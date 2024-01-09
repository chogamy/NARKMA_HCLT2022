import os

import torch
from numpy import average
import matplotlib.pyplot as plt
import numpy
# from mytokenizer import MyTokenizer

DIR = os.path.dirname(os.path.realpath(__file__))

class datas:
    def __init__(self) -> None:
        self.beams0 = []
        self.beams1 = []
        self.beams2 = []
        self.beams3 = []
        self.beams4 = []

        self.tgts = []

        self.srcs = []
        self.morphs = []
        self.tags = []
        self.tgt_morphs = []
        self.tgt_tags = []

        self.F1 = []
        self.Acc = []
        self.total_len_dif = []
        self.tag_len_dif = []

        def load():
            beams0 = open(DIR + "/inf/beam0.txt", 'r', encoding="utf-8-sig")
            beams1 = open(DIR + "/inf/beam1.txt", 'r', encoding="utf-8-sig")
            beams2 = open(DIR + "/inf/beam2.txt", 'r', encoding="utf-8-sig")
            beams3 = open(DIR + "/inf/beam3.txt", 'r', encoding="utf-8-sig")
            beams4 = open(DIR + "/inf/beam4.txt", 'r', encoding="utf-8-sig")
            tgts = open(DIR + "/test_tgt.txt", 'r', encoding="utf-8-sig")
            for beam0, beam1, beam2, beam3, beam4, tgt in zip(beams0, beams1, beams2, beams3, beams4, tgts):
                self.beams0.append(beam0.strip())
                self.beams1.append(beam1.strip())
                self.beams2.append(beam2.strip())
                self.beams3.append(beam3.strip())
                self.beams4.append(beam4.strip())

                self.tgts.append(tgt.strip())
            srcs = open(DIR + "/test_src.txt", 'r', encoding="utf-8-sig")
            morphs = open(DIR + "/inf/morph.txt", 'r', encoding="utf-8-sig")
            tags = open(DIR + "/inf/tag.txt", 'r', encoding="utf-8-sig")
            tgt_morphs = open(DIR + "/test_morph.txt", 'r', encoding="utf-8-sig")
            tgt_tags = open(DIR + "/test_tag.txt", 'r', encoding="utf-8-sig")
            for src, morph, tag, tgt_morph, tgt_tag in zip(srcs, morphs, tags, tgt_morphs, tgt_tags):
                self.srcs.append(src.strip())
                self.morphs.append(morph.strip())
                self.tags.append(tag.strip())
                self.tgt_morphs.append(tgt_morph.strip())
                self.tgt_tags.append(tgt_tag.strip())
        load()

        for inf, tgt in zip(self.beams0, self.tgts):
            self.F1.append(self.f1(inf, tgt))
            self.Acc.append(self.acc(inf, tgt))
    
        
        def len_dif():           
            for inf_tag ,tgt_tag in zip(self.tags, self.tgt_tags):
                inf_tag = inf_tag.split(" ")
                tgt_tag = tgt_tag.split(" ")
                inf_len = len(inf_tag)
                tgt_len = len(tgt_tag)
                self.tag_len_dif.append(inf_len - tgt_len)
        len_dif()
        
        def encode_len_dif():
            for inf ,tgt in zip(self.beams0, self.tgts):
                inf = self.encode(inf)
                tgt = self.encode(tgt)
                
                inf_len = len(inf)
                tgt_len = len(tgt)
                self.total_len_dif.append(inf_len - tgt_len)
        encode_len_dif()

        self.dist("F1",self.F1)
        self.dist("ACC",self.Acc)

        self.err()
        self.rep_sent()

        self.len_err()

    def f1(self, infs, tgts):
        infs = infs.replace("+", " ").split(" ")
        tgts = tgts.replace("+", " ").split(" ")

        result = set()
        target = set()

        for tgt in tgts:
            k = 0
            while (1):
                if (tgt, k) in target:
                    k += 1
                else :
                    target.add((tgt,k))
                    break
        
        for inf in infs:
            k = 0
            while (1):
                if (inf, k) in result:
                    k += 1
                else :
                    result.add((inf,k))
                    break
        
        answer = result & target

        p = len(answer) / len(result)
        r = len(answer) / len(target)

        f1 = 2 * p * r / (p + r) if p + r != 0 else 0

        return f1

    def tag_f1(self, infs, tgts):
        infs = infs.split(" ")
        infs = "".join(infs)
        tgts = tgts.split(" ")
        tgts = "".join(tgts)

        infs = infs.replace("/O+", "/O").split("/O")
        tgts = tgts.replace("/O+", "/O").split("/O")

        result = set()
        target = set()

        for tgt in tgts:
            k = 0
            while (1):
                if (tgt, k) in target:
                    k += 1
                else :
                    target.add((tgt,k))
                    break
        
        for inf in infs:
            k = 0
            while (1):
                if (inf, k) in result:
                    k += 1
                else :
                    result.add((inf,k))
                    break
        
        answer = result & target

        p = len(answer) / len(result)
        r = len(answer) / len(target)

        f1 = 2 * p * r / (p + r) if p + r != 0 else 0

        return f1

    def acc(self, infs, tgts):
        infs = infs.split(" ")
        tgts = tgts.split(" ")

        result = set()
        target = set()

        for tgt in tgts:
            k = 0
            while (1):
                if (tgt, k) in target:
                    k += 1
                else :
                    target.add((tgt,k))
                    break
        
        for inf in infs:
            k = 0
            while (1):
                if (inf, k) in result:
                    k += 1
                else :
                    result.add((inf,k))
                    break
        
        answer = result & target

        p = len(answer) / len(result)
        r = len(answer) / len(target)

        acc = p if p != 0 else 0

        return acc

    def tag_acc(self, infs, tgts):
        infs = infs.split(" ")
        infs = "".join(infs)
        tgts = tgts.split(" ")
        tgts = "".join(tgts)

        infs = infs.split("/O")
        tgts = tgts.split("/O")

        result = set()
        target = set()

        for tgt in tgts:
            k = 0
            while (1):
                if (tgt, k) in target:
                    k += 1
                else :
                    target.add((tgt,k))
                    break
        
        for inf in infs:
            k = 0
            while (1):
                if (inf, k) in result:
                    k += 1
                else :
                    result.add((inf,k))
                    break
        
        answer = result & target

        p = len(answer) / len(result)
        r = len(answer) / len(target)

        acc = p if p != 0 else 0

        return acc
   
    # def err(self):
    #     count = 0
    #     scores = self.Acc
    #     f = open(f"{DIR}/err.txt", "w", encoding="utf-8-sig")
        
    #     for i, score in enumerate(scores):
    #         if score <= 0.1 :
    #             self.writer(f, i, self.beams0[i], self.tgts[i])
    #             count+=1
    #     print("ERR : ", count)

    # def err(self):
    #     count = 0
    #     f = open(f"{DIR}/err.txt", "w", encoding="utf-8-sig")

    #     for i in range(len(self.beams0)):
    #         if "/O" in self.beams0[i]:
    #             count+=1
    #             self.writer(f, i, self.beams0[i], self.tgts[i], None)
    #     print("/O가 있는 ERR : ", count)

    def err(self):
        count = 0
        f = open(f"{DIR}/err.txt", "w", encoding="utf-8-sig")
        count = 0
        for i in range(len(self.beams0)):
            if self.beams0[i] != self.tgts[i]:
                count += 1
        print("EM 아닌 문장 : ", count)

        count = 0
        f1s = []
        accs = []
        for i in range(len(self.beams0)):
            src_eojs = self.beams0[i].split(" ")
            tgt_eojs = self.tgts[i].split(" ")
            if len(src_eojs) != len(tgt_eojs):
                f1 = self.f1(self.beams0[i], self.tgts[i])
                acc = self.acc(self.beams0[i], self.tgts[i])
                f1s.append(f1)
                accs.append(acc)
                print(f"어절 차이 {len(src_eojs)} {len(tgt_eojs)}")
                print(f"f1 : {f1}, acc : {acc}")
                if len(tgt_eojs) < 10:
                    print(f"{i}번째 문장")
                    print(self.beams0[i])
                    print(self.tgts[i])
                f.write(f"{i}번째 문장 : \n")
                f.write(self.beams0[i])
                f.write("\n")
                f.write(self.tgts[i])
                f.write("\n")
                count+=1
        print("어절 갯수가 다른 문장 : ", count)
        print(f"걔네 어버리지 f1 {average(f1s)}")
        print(f"걔네 어버리지 acc {average(accs)}")

    def len_err(self) : 
        '''
        합친 문장 말고. 
        형태소 or 태그로 길이 차이 확인해야 할듯
        '''
        count = 0
        for i in range(len(self.beams0)):
            inf = self.encode(self.beams0[i])
            inf_len = len(inf)

            tgt = self.encode(self.tgts[i])
            tgt_len = len(tgt)

            if inf_len != tgt_len:
                count +=1
        #print(f"len 차이나는 문장 : {count}")

    def encode(self, sent):
        ''''
        !!!!!@@@@@ DONT EVER NEVER CHANGE!!!!!!@@@@@@@@@@@@
        !!!!! CHANGE IN EVAL TOKEN REPETITION!!!@@@@@@@@@
        '''
        result = []

        eojeols = sent.split(" ")
        for eojeol in eojeols:
            morphs = eojeol.split("+")
            for morph in morphs:
                if morph == "":
                    result.append("+")  # 엥 이거 그냥 없애버리는게 좋은거 아니야?
                    continue
                if len(morph.rsplit("/", 1)) == 1:
                    syls = morph.rsplit("/", 1)
                    tag = ""
                else :
                    syls, tag = morph.rsplit("/", 1)
                    # 오로지 음절만 있는지 확인
                    # if syls != "/" and "/" in syls:
                    #     print(syls)
                    tag = "/"+tag
                for i in range(len(syls)):
                    result.append(syls[i])
                if tag == "":
                    pass
                else :
                    result.append(tag)
                result.append("+")
            result.pop()
            result.append(" ")
        result.pop()
     
        assert "".join(result) == sent, f"{sent}"
    
        return result
    
    def rep_check(self, encode_list):
        for i in range(len(encode_list) - 1):
            if encode_list[i] == encode_list[i+1]:
                return True
        
        return False

    def rep_list(self, encode_list):
        result = []
        for i in range(len(encode_list) - 1):
            if encode_list[i] == encode_list[i+1]:
                rep = str(encode_list[i]) + str(encode_list[i+1])
                result.append(rep)
        return result
    
    def rep_sent(self):
        f = open(DIR + "/rep_chekc.txt", 'w', encoding="utf-8-sig")
        count = 0
        for i in range(len(self.beams0)):
            inf = self.encode(self.beams0[i])
            tgt = self.encode(self.tgts[i])
            if self.rep_check(inf) and not self.rep_check(tgt):
                rep_list = self.rep_list(inf)
                self.writer(f, i, self.beams0[i], self.tgts[i], "  ".join(rep_list))
                count +=1 
        print(f"반복 문장 : {count}")

    def rep_counter(self, sents):
        rep = 0
        for sent in sents:
            encode_list = self.encode(sent)
            for i in range(len(encode_list) - 1):
                if encode_list[i] == encode_list[i+1]:
                    rep += 1
        return rep

    def dist(self, name, list):
        bins = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        #print(f"{name} dist")
        frq, bins, fig = plt.hist(list, bins=bins)
        frq = [int(x) for x in frq]
        plt.ylabel("freq", fontsize=10)
        plt.xlabel(f"{name}",fontsize=10)
        plt.xticks(bins)
        plt.savefig(f"{name}.png")
        plt.grid()
        plt.show()
        plt.clf()
        #print(f"freq : {frq}")
        #print(f"{name} : {bins}")

    def space_check(self):
        less_count = 0
        more_count = 0
        for inf, tgt in zip(self.beams0, self.tgts):
            inf_space = inf.split(" ")
            tgt_space = tgt.split(" ")
            if len(inf_space) > len(tgt_space):
                more_count += 1
            elif len(inf_space) < len(tgt_space):
                less_count += 1
        print(f"inf가 어절 더 생성: {more_count}")
        print(f"inf가 어절 덜 생성: {less_count}")
            
    def writer(self, file ,index, inf, tgt, others=None):
        file.write(f"{index}번째 문장 : F1 - {self.F1[index]} // ACC : {self.Acc[index]}\n")
        file.write(f"inf : {inf}\n")
        file.write(f"src : {self.srcs[index]}\n")
        file.write(f"tgt : {tgt}\n")
        if others : 
            file.write(f"{others}")
        file.write("\n-----------------\n")

    
test =datas()

print(f"F1  : {average(test.F1)}")
print(f"Acc : {average(test.Acc)}")

morph_f1 = []
morph_acc = []
for inf, tgt in zip(test.morphs, test.tgt_morphs):
    morph_f1.append(test.f1(inf, tgt))
    morph_acc.append(test.acc(inf,tgt))

tag_f1 = []
tag_acc = []
for inf, tgt in zip(test.tags, test.tgt_tags):
    tag_f1.append(test.tag_f1(inf, tgt))
    tag_acc.append(test.tag_acc(inf,tgt))
    

print(f"morph f1 : {average(morph_f1)}  // morph acc : {average(morph_acc)} ")
print(f"tag f1 :  {average(tag_f1)} // tag acc : {average(tag_acc)} ")

test.space_check()

print("전체 문장에 대한 길이차이")
print(f"max Len dif : {max(test.total_len_dif)}")
print(f"min Len dif : {min(test.total_len_dif)}")
abs_len =[abs(x) for x in test.total_len_dif]
print(f"avg abs dif : {average(abs_len)}")


print("태그에 대한 길이차이 - 이것이 리얼")
print(f"max Len dif : {max(test.tag_len_dif)}")
print(f"min Len dif : {min(test.tag_len_dif)}")
abs_len =[abs(x) for x in test.tag_len_dif]
print(f"avg abs dif : {average(abs_len)}")




# 길이 차이 나는 문장 max/min
# max_index = numpy.argmax(test.Len)
# min_index = numpy.argmin(test.Len)
# print(f"{max_index} 문장 : F1 - {test.F1[max_index]} // ACC - {test.Acc[max_index]}")
# print(f"max inf : {test.beams0[max_index]}")
# print("--------------------")
# print(f"max tgt : {test.tgts[max_index]}")
# print("--------------------\n")
# print(f"{min_index} 문장 : F1 - {test.F1[min_index]} // ACC - {test.Acc[min_index]}")
# print(f"min inf : {test.beams0[min_index]}")
# print("--------------------")
# print(f"min tgt : {test.tgts[min_index]}")



morph_rep = 0
tgt_morph_rep = 0
for inf, tgt in zip(test.morphs, test.tgt_morphs):
    for i in range(len(inf) - 1):
        if inf[i] == inf[i+1]:
            morph_rep += 1
    for i in range(len(tgt) - 1):
        if tgt[i] == tgt[i+1]:
            tgt_morph_rep += 1

print(f"morph rep : {morph_rep} {tgt_morph_rep} // 차이 : {morph_rep - tgt_morph_rep}")

tag_rep = 0
tgt_tag_rep = 0
for inf, tgt in zip(test.tags, test.tgt_tags):
    inf = inf.split(" ")
    tgt = tgt.split(" ")
    for i in range(len(inf) - 1):
        if inf[i] == inf[i+1]:
            tag_rep += 1
    for i in range(len(tgt) - 1):
        if tgt[i] == tgt[i+1]:
            tgt_tag_rep += 1

print(f"tag rep : {tag_rep} {tgt_tag_rep}  // 차이 : {tag_rep - tgt_tag_rep}")




print(f"inf 토큰 반복 갯수 :  {test.rep_counter(test.beams0)}   //차이 : {test.rep_counter(test.beams0) - test.rep_counter(test.tgts)}")
print(f"tgt 토큰 반복 갯수 :  {test.rep_counter(test.tgts)}")