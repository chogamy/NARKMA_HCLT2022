import os

class FileManager():
    def __init__(self, path = os.path.dirname(os.path.abspath(__file__)), file_name_list=["beam"], beam_size = 5) -> None:
        self.path = path + "/inf/"
        self.beam_size=beam_size
        self.file_name_list = file_name_list

        self.file_list = []

        self.beam_file_list = []
        self.BI_beam_file_list = []
        self.prob_file_list = []

        self.file_open()
    
    def file_open(self):
        for file_name in self.file_name_list:
            for i in range(self.beam_size):
                if file_name == "beam":
                    file_buf = open(self.path + file_name + str(i) + ".txt", 'w', encoding="utf-8-sig")
                    #file_bi_buf = open(self.path + "BI_" + file_name + str(i) + ".txt", 'w', encoding="utf-8-sig")
                    self.beam_file_list.append(file_buf)
                    #self.BI_beam_file_list.append(file_bi_buf)
                elif file_name == "prob" :
                    file_buf = open(self.path + file_name + str(i) + ".txt", 'w', encoding="utf-8-sig")
                    self.prob_file_list.append(file_buf)
                else :
                    pass
        for f in self.beam_file_list:
            self.file_list.append(f)
        for f in self.BI_beam_file_list:
            self.file_list.append(f)
        for f in self.prob_file_list:
            self.file_list.append(f)            

    def file_close(self):
        for file in self.file_list:
            file.close()

    def make_sentence(self, morphs, tags):
        sentence =[]
        for morph, tag in zip(morphs, tags):
            if morph != "<pad>" and tag != "<pad>":
                sentence.append(morph)
                sentence.append(tag)
            else :
                pass
        return sentence

    def sent_to_beam_write(self, sent, i):
        self.beam_file_list[i].write(sent)
        self.beam_file_list[i].write("\n")

    def beam_write(self, morphs, BIs, i):
        #tags = self.BI_to_tag(BIs)
        tags = self.force_BI_to_tag(BIs)
        sentence = self.make_sentence(morphs, tags)
        sentence = "".join(sentence)
        self.beam_file_list[i].write(sentence)
        self.beam_file_list[i].write("\n")

    def BI_write(self, morphs, BIs, i):
        sentence = self.make_sentence(morphs, BIs)
        sentence = " ".join(sentence)
        self.BI_beam_file_list[i].write(sentence)
        self.BI_beam_file_list[i].write("\n")
    
    def prob_write(self):
        pass
    
    def No_BI_last(self, morph, tag):
        result = []
        result.append(morph[0])
        for i in range(1, len(morph)):
            if morph[i] == "+" or morph[i] == " ":
                if tag[i-1] == "/O" or tag[i-1] == "/O+":
                    result.append("")
                else :
                    result.append(tag[i-1])
            result.append(morph[i])

        if tag[-1] != "/O" and tag[-1] != "/O+":
            result.append(tag[-1])
        return "".join(result)
                    
    # def No_BI_first(self, morph, tag):
    #     result = []
    #     cur_tag = tag[0]

    #     if cur_tag == "/O+" or cur_tag == "/O":
    #         cur_tag = ""

    #     for i in range(len(morph) - 1):
    #         if morph[i] == "+" or morph[i] == " ":
    #             result.append(cur_tag)
    #             cur_tag = tag[i+1]
                
    #             if cur_tag == "/O+" or cur_tag == "/O":
    #                 cur_tag = ""
                
    #         result.append(morph[i])
        
    #     if len(morph) == 1:
    #         return "".join(result)

    #     # for last
    #     result.append(morph[-1])
    #     if cur_tag == "/O+" or cur_tag == "/O":
    #         cur_tag = ""
    #     result.append(cur_tag)

    #     return "".join(result)

    def No_BI_first(self, morph, tag):
        result = []

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
            
    def BI_to_tag_first(self, morph, BI):
        result = []
        result.append(morph[0])

        cur_tag = BI[0]
        if cur_tag == "/O" or cur_tag == "/O+":
            cur_tag = BI[1]

        for i in range(1, len(morph)-1):
            if morph[i] == "+" or morph[i] == " ":
                cur_tag = cur_tag.replace("B-", "")
                cur_tag = cur_tag.replace("I-", "")
                result.append(cur_tag)
                cur_tag = BI[i+1]
            result.append(morph[i])
        
        result.append(morph[-1])
        if cur_tag != "/O" and cur_tag != "/O+":
            cur_tag = cur_tag.replace("B-", "")
            cur_tag = cur_tag.replace("I-", "")
            result.append(cur_tag)
        return "".join(result)

    def force_BI_to_tag(self, BI : list) :
        result_tag = []
        bi = BI[:]
        
        cur_tag = bi.pop(0)
        for comp_tag in bi:
            if "I-" in comp_tag:
                if cur_tag == "/O" or cur_tag == "/O+":
                    # Non-sense
                    # /O,   /I- ~
                    # /O+,  /I- ~
                    result_tag.append("")
                    cur_tag = comp_tag.replace("I-", "") 
                else : 
                    result_tag.append("")
            else:
                result_tag.append(cur_tag.replace("B-", ""))
                cur_tag = comp_tag
        result_tag.append(cur_tag.replace("B-", ""))

        for i in range(len(result_tag)):
            if result_tag[i] == "/O+" or result_tag[i] == "/O":
                result_tag[i] = ""
        
        assert len(result_tag) == len(BI), f"len difference\n{result_tag}\n{BI}"
        return result_tag
