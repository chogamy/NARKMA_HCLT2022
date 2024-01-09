import os

import torch
from numpy import average
import matplotlib.pyplot as plt
import numpy

DIR = os.path.dirname(os.path.realpath(__file__))

print("inf0 DIR : ")
#INF0_DIR = input()
INF0_DIR = "/root/zHCLT_NAR_KMA_1layer_m-t/inf_0"

print("inf1 DIR : ")
#INF1_DIR = input()
INF1_DIR = "/root/zVis_2layer/inf0"

print("inf2 DIR : ")
#INF2_DIR = input()
INF2_DIR = "/root/NAR_KMA_BASELINE_ONE_layer/inf"


class datas:
    def __init__(self) -> None:
        self.infs0 = []
        self.infs1 = []
        self.infs2 = []
        

        self.tgts = []

        self.srcs = []
        self.morphs = []
        self.tags = []
        self.tgt_morphs = []
        self.tgt_tags = []


        def load():
            infs0 = open(INF0_DIR + "/beam0.txt", 'r', encoding="utf-8-sig")
            infs1 = open(INF1_DIR + "/beam0.txt", 'r', encoding="utf-8-sig")
            tgts = open(DIR + "/test_tgt.txt", 'r', encoding="utf-8-sig")
            for inf0, inf1, tgt in zip(infs0, infs1, tgts):
                self.infs0.append(inf0.strip())
                self.infs1.append(inf1.strip())
                self.tgts.append(tgt.strip())

            srcs = open(DIR + "/test_src.txt", 'r', encoding="utf-8-sig")
            for src in srcs:
                self.srcs.append(src.strip())
                
        load()
    
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

    def tag_f1(self, inf, tgt):
        inf = inf.split(" ")
        inf = "".join(inf)
        tgt = tgt.split(" ")
        tgt = "".join(tgt)

        inf = inf.replace("/O+", "/O").split("/O")
        tgt = tgt.replace("/O+", "/O").split("/O")

        inf = set(inf)
        tgt = set(tgt)

        ans = inf & tgt

        prec = len(ans) / len(inf)
        recall = len(ans) / len(tgt)

        if (prec + recall) == 0:
            f1 = 0
        else : 
            f1 = 2 * prec * recall / (prec + recall)
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

    def tag_acc(self, inf, tgt):
        inf = inf.split(" ")
        inf = "".join(inf)
        tgt = tgt.split(" ")
        tgt = "".join(tgt)

        inf = inf.split("/O")
        tgt = tgt.split("/O")

        inf = set(inf)
        tgt = set(tgt)

        ans = inf & tgt

        acc = len(ans) / len(inf)

        return acc
    
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
    
    
    
    def rep_check(self, sent):
        encode_list = test.encode(sent)
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
            
    def writer(self, file ,index):
        file.write(f"{index}번째 문장 \n")
        file.write(f"src : {self.srcs[index]}\n")
        file.write(f"DDAN 적용전  : {self.infs1[index]}\n")
        file.write(f"f1 : {self.f1(self.infs1[index], self.tgts[index])} //  ACC : {self.acc(self.infs1[index], self.tgts[index])} \n")
        file.write(f"DDAN 적용후 : {self.infs0[index]} \n")
        file.write(f"f1 : {self.f1(self.infs0[index], self.tgts[index])} //  ACC : {self.acc(self.infs0[index], self.tgts[index])} \n")
        file.write(f"정답 : {self.tgts[index]}\n")
        file.write("-----------------\n")
        

    def what_to_show_rep(self):
        f = open(DIR + "/comp.txt", 'w', encoding="utf-8-sig")

        count = 0
        for i in range(len(self.infs1)):
    
            a = self.rep_check(self.infs0[i])   # ddan 적용
            b = self.rep_check(self.infs1[i])   # ddan 미적용
            c = self.rep_check(self.tgts[i])

            if (not a and b) and not c: # ddan 미적용 반복 문장
                a_f1 = self.f1(self.infs0[i], self.tgts[i])
                a_acc = self.acc(self.infs0[i], self.tgts[i])

                b_f1 = self.f1(self.infs1[i], self.tgts[i])
                b_acc = self.acc(self.infs1[i], self.tgts[i])

                if a_f1 >= b_f1 and a_acc >= b_acc :
                    count+=1
                    self.writer(f, i)
                

        print(count)

        f.close()

    def what_to_show_Otag(self):
        f = open(DIR + "/comp_Otag.txt", 'w', encoding="utf-8-sig")

        count = 0
        for i in range(len(self.infs1)):
    
            a = True if "/O" in self.infs0[i] else False
            b = True if "/O" in self.infs1[i] else False
            c = False

            if a and not b:
                a_f1 = self.f1(self.infs0[i], self.tgts[i])
                a_acc = self.acc(self.infs0[i], self.tgts[i])

                b_f1 = self.f1(self.infs1[i], self.tgts[i])
                b_acc = self.acc(self.infs1[i], self.tgts[i])
                if b_f1 >= a_f1 and b_acc >= a_acc:
                    count+=1
                    #other1 = f"con : {self.infs1[i]}  \n  \t f1 : {b_f1} //  acc : {b_acc}"
                    self.writer(f, i)
                

        print(count)

        f.close()

    
    def what_to_show_correct(self):
        # f = open(DIR + "/comp_corr.txt", 'w', encoding="utf-8-sig")
        count = 0
        for i in range(len(self.infs0)):
            
            a = True if self.infs0[i] == self.tgts[i] else False
            b = True if self.infs1[i] == self.tgts[i] else False


            if a and not b and len(self.tgts[i]) < 30:
                print(i)
                print("hclt : ")
                print(self.infs0[i])
                print("2층 : ")
                print(self.infs1[i])
                print("-----------------------------")
                count += 1
        print(count)


test =datas()
test.what_to_show_correct()


