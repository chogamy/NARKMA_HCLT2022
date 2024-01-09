import os
import argparse
from tqdm import tqdm
import yaml
import math

import torch
import torch.nn.functional as F

from numpy import average

from mytokenizer import MyTokenizer
from nat_base import _expand_mask
from train_nat import Model


from splitter_inf import split


def duplicate_encoder_out(encoder_out, att_mask, bsz, beam_size):
    new_encoder_out = encoder_out.unsqueeze(2).repeat(beam_size, 1, 1, 1).view(bsz * beam_size, encoder_out.size(1), -1)
    new_att_mask = att_mask.unsqueeze(1).repeat(beam_size, 1, 1).view(bsz * beam_size, -1)
    return new_encoder_out, new_att_mask

def predict_length_beam(predicted_lengths, length_beam_size):
    beam_probs = predicted_lengths.topk(length_beam_size, dim=1)[0]
    beam = predicted_lengths.topk(length_beam_size, dim=1)[1]
    beam = beam[0].tolist()
    beam_probs = beam_probs[0].tolist()
    return beam, beam_probs

def make_enc_input(input_ids, tok, max_len):
    attention_mask = [1] * len(input_ids) \
                        + [0] * (max_len - len(input_ids))
    input_ids = input_ids + [tok.index("<pad>")] * (max_len - len(input_ids))

    return input_ids, attention_mask

def make_dec_input(length, max_len, tok):
    decoder_input_ids = [tok.index("<mask>")] * length \
                        + [tok.index("<pad>")] * (max_len - length)

    decoder_attention_mask = [1] * length \
                        + [0] * (max_len - length)

    return decoder_input_ids, decoder_attention_mask

def argmax(logits):
    '''
    logits : beamsize * length * Vocab_size
    --> argmax : beamsize * length
    '''
    # logits : beamsize * length * Vocab_size
    probs = F.softmax(logits, dim=-1)
    max_probs, idx = probs.max(dim=-1)
    return idx, max_probs, probs

def length_predictor(length_logit, min_len):
    length_logit[:, :min_len] += float('-inf')              # src len이 target보다 작을수가 없다.
    length_probs = F.log_softmax(length_logit, dim=-1)      # length.size() = 1 * Maxlen. 각각의 로그 확률
    length_cands, length_probs = predict_length_beam(length_logit, args.length_beam_size)  
    return length_cands, length_probs

def inference(model, sent, src_tok, args):
    source_len = len(list(sent))
    input_ids = src_tok.encode(list(sent)) 
    input_ids.insert(0, src_tok.index("<len>"))
    
    input_ids, attention_mask = make_enc_input(input_ids, src_tok, args.max_len)

    attention_mask = torch.tensor(attention_mask)
    attention_mask = attention_mask.unsqueeze(0)
    attention_mask = attention_mask.cuda()

    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.cuda()

    enc_outputs, length = model.encoder(input_ids, attention_mask)

    # LENGTH
    length_cands, length_probs = length_predictor(length, source_len)

    dec_inputs = []
    dec_attention_masks = []
    for len_can in length_cands:
        #dec_input, dec_att_mask = make_dec_input(input_ids[1:], len_can, morph_tok, max_len, src_tok.index("<pad>"))
        dec_input_ids, dec_attention_mask = make_dec_input(len_can, args.max_len, morph_tok)
        dec_inputs.append(dec_input_ids)
        dec_attention_masks.append(dec_attention_mask)

    dec_attention_masks = torch.tensor(dec_attention_masks)
    dec_attention_masks = dec_attention_masks.cuda()
    dec_inputs = torch.tensor(dec_inputs)
    dec_inputs = dec_inputs.cuda()

    enc_outputs, attention_mask = duplicate_encoder_out(enc_outputs, attention_mask, enc_outputs.size(0), args.length_beam_size)
    
    morph_outputs, _ = model.morph_decoder(dec_inputs, dec_attention_masks, 
                                            enc_outputs, attention_mask[:, 1:])

    tag_outputs, _ = model.tag_decoder(dec_inputs, dec_attention_masks, 
                                        enc_outputs, attention_mask[:, 1:])

    
    dec_attention_masks = _expand_mask(dec_attention_masks, morph_outputs.dtype)

    morph_outputs, _ = model.morph_decoder.layers[0](
                morph_outputs,
                tag_outputs,
                dec_attention_masks,
                dec_attention_masks
        )
    
    tag_outputs, _ = model.tag_decoder.layers[0](
                tag_outputs,
                morph_outputs,
                dec_attention_masks,
                dec_attention_masks
        )

    morph_logits = model.morph_projection(morph_outputs)
    tag_logits = model.tag_projection(tag_outputs)

    morph_ids, morph_probs, _ = argmax(morph_logits)   
    for i in range(args.length_beam_size):
        morph_ids[i][length_cands[i]:] = morph_tok.pad()
        morph_probs[i][length_cands[i]:] = 1

    tag_ids, tag_probs, _ = argmax(tag_logits)
    for i in range(args.length_beam_size):
        tag_ids[i][length_cands[i]:] = tag_tok.pad()
        tag_probs[i][length_cands[i]:] = 1
    
    beam_ids, beam_probs = choose_beam(morph_probs, tag_probs, length_probs, length_cands, args)

    return morph_ids, tag_ids, morph_probs, tag_probs, beam_ids, beam_probs, length_cands
           
def choose_beam(morph_probs, tag_probs, length_probs, length_cands, args):
    morph_lprobs = morph_probs.log().sum(-1)
    tag_lprobs = tag_probs.log().sum(-1) 
    length_probs = torch.tensor(length_probs).cuda() * 0.1  # length reflection ratio

    beam_score = (tag_lprobs + morph_lprobs + length_probs) / torch.tensor(length_cands).cuda() 
    
    beam_probs, beam_ids = beam_score.topk(args.length_beam_size)
    
    return beam_ids.tolist(), beam_probs.tolist()

def decoding(morph_tok, tag_tok, morph_ids, tag_ids, beam_ids, length_cands, args):
    '''
    beams * ids -> beams * tokens
    '''
    morph_beam = []
    tag_beam = []
    
    for i in range(args.length_beam_size):
        assert len(morph_ids[beam_ids[i]].tolist()) == len(tag_ids[beam_ids[i]].tolist()), f"ids length different"
        morph_result = morph_tok.decode(morph_ids[beam_ids[i]].tolist(), False)
        tag_result = tag_tok.decode(tag_ids[beam_ids[i]].tolist(), False)

        length = length_cands[beam_ids[i]]
        morph_result = morph_result[:length]
        tag_result = tag_result[:length]

        
        morph_beam.append("".join(morph_result))
        tag_beam.append(" ".join(tag_result))
        
    return morph_beam, tag_beam

def unite(morphs, tags):
    tags = tags.split(" ")
    tag_pointer = 0
    eojeols = morphs.split(" ")
    result = []
    print(morphs)
    for eojeol in eojeols:
        morphemes = eojeol.split("+")
        morpheme_result = []
        for i in range(len(morphemes)): # 0이면 안돌아
            morph_tag = morphemes[i] + tags[tag_pointer]
            morpheme_result.append(morph_tag)
            tag_pointer += len(list(morphemes[i]))
            if morphemes[i] != morphemes[-1]: # 마지막 형태소가 아니면 +1
                tag_pointer+=1
        tag_pointer += 1
        
        morpheme_result = "+".join(morpheme_result)
        result.append(morpheme_result)
    result = " ".join(result)
    return result

def No_BI_first(morph, tag):
    result = []
    tag = tag.split(" ")

    cur_tag = tag[0]
    
    morph = list(morph)
    morph.append("<end>")
    tag.append("<end>")
    for i in range(len(morph)-1):
        result.append(morph[i])

        # syl+    or syl" "
        if (morph[i] != "+" and morph[i] != " ") and (morph[i+1] == "+" or morph[i+1] == " "):
            result.append(cur_tag) 
            cur_tag = ""
        # " "syl or +syl
        elif (morph[i] == " " or morph[i] == "+") and (morph[i+1] != "+" and morph[i+1] != " "):
            cur_tag = tag[i+1]
        # ++ or (+ ) or ( +) or (  )
        elif (morph[i] == "+" or morph[i] == " ") and (morph[i+1] == "+" or morph[i+1] == " "):
            result.pop()
        # syl, syl?
    
    if cur_tag != "<end>" and result[-1] != "/O" and result[-1] != "/O+":
        result.append(cur_tag)
        
    return "".join(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default=None, type=str)
    parser.add_argument("--model_binary", default=None, type=str)
    parser.add_argument("--testfile", default=None, type=str)
    parser.add_argument("--outputfile", default=None, type=str)
    parser.add_argument("--gold_len", default=False, type=bool)
    parser.add_argument("--length_beam_size", default=3, type=int)
    args = parser.parse_args()

    with open(args.hparams) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
        hparams.update(vars(args))

    args = argparse.Namespace(**hparams)

    inf = Model.load_from_checkpoint(args.model_binary, args=args)
    model = inf.model
    model = model.cuda() 
    model.eval()

    src_tok = inf.src_tok
    morph_tok = inf.morph_tok
    tag_tok = inf.tag_tok

    assert morph_tok.index("<mask>") == tag_tok.index("<mask>"), "mask index different"

    #  input 입력 문장
    srcs = []
    f = open(args.testfile + '_src.txt', 'r', encoding="utf-8-sig")
    for src in f:
        srcs.append(src.strip())
    f.close()


    infs_morph = []
    infs_tag = []
    infs = [] # splitted sent의 morph, tag를 한번에.
    results = []

    for src in tqdm(srcs, total=len(srcs)):
        morph_beam = []
        tag_beam = []
        for i in range(args.length_beam_size):
            morph_beam.append([])
            tag_beam.append([])

        sents = split(src, args.max_len)
        result_per_sent = []
        for sent in sents:
            morph_ids, tag_ids, morph_probs, tag_probs, beam_ids, beam_probs, length_cands = inference(model, sent, src_tok, args)      

            morph_buffer_beam, tag_buffer_beam = decoding(morph_tok, tag_tok, morph_ids, tag_ids, beam_ids, length_cands, args)          
            
            for i in range(args.length_beam_size):
                morph_beam[i].append(morph_buffer_beam[i])
                tag_beam[i].append(tag_buffer_beam[i])
            result_buffer = No_BI_first(morph_buffer_beam[0], tag_buffer_beam[0])
            #result_buffer = unite(morph_buffer_beam[0], tag_buffer_beam[0])
            result_buffer = result_buffer.replace("/O+", "")
            result_buffer = result_buffer.replace("/O", "")
            result_buffer = result_buffer.replace("++++", "+")
            result_buffer = result_buffer.replace("+++", "+")
            result_buffer = result_buffer.replace("++", "+")
            result_per_sent.append(result_buffer)
        results.append("".join(result_per_sent))

        morph_result = "".join(morph_beam[0])
        tag_result = " ".join(tag_beam[0])

        #result = unite(morph_result, tag_result)
        #result = No_BI_first(morph_result, tag_result)
        infs_morph.append(morph_result)
        infs_tag.append(tag_result)
        # infs.append(result)


    DIR = os.path.dirname(os.path.realpath(__file__)) + "/inf/"

    morph_file = open(DIR + "morph.txt", 'w', encoding="utf-8-sig")
    tag_file = open(DIR + "tag.txt", 'w', encoding="utf-8-sig")
    beam0 = open(DIR + "beam0.txt", 'w', encoding="utf-8-sig")

    for inf in results:
    #for inf in infs:
        beam0.write(inf.strip())
        beam0.write("\n")

    for morph in infs_morph:
        morph_file.write(morph)
        morph_file.write("\n")
    
    for tag in infs_tag:
        tag_file.write(tag)
        tag_file.write("\n")

    morph_file.close()
    tag_file.close()
    beam0.close()