def last(src, max_len):

    src_bufs = src.rsplit(" ")
    
    for i, src_buf in enumerate(src_bufs):
        if src_bufs[-1] != src_buf:
            src_bufs[i] += " "




    src_bufs2 = []
    src_result = []

    for src_buf in src_bufs:
        src_bufs2.append(src_buf)
    
        if len("".join(src_bufs2)) > (max_len / 2):
            src_result.append("".join(src_bufs2))
            src_bufs2 = []

    src_result.append("".join(src_bufs2))

    src_bufs = src_result

    if src_bufs[-1] == "":
        src_bufs.pop()
    val = "".join(src_bufs)
    assert "".join(src_bufs) == src, f"공백 잘 못 짜른듯...\n{val}\n{src}"
    return src_bufs



def left_token(src, src_token):
    src_bufs = src.rsplit(src_token)
    
    for i, src_buf in enumerate(src_bufs):
        if src_bufs[0] != src_buf:
            src_bufs[i] = src_token + src_bufs[i]

    if src_bufs[0] == "" :
        src_bufs.pop(0)
    assert "".join(src_bufs) == src, f"잘 못 짜른듯..."
    return src_bufs


def token(src, src_token):
    src_bufs = src.rsplit(src_token)
    
    for i, src_buf in enumerate(src_bufs):
        if src_bufs[-1] != src_buf:
            src_bufs[i] += src_token
    assert "".join(src_bufs) == src, f"잘 못 짜른듯..."

    return src_bufs


def split(src, max_len) : 
    original_sent = src
    src_spl = []

    org_max_len = max_len
    max_len = max_len * 4 / 5
    
    if len(src) < max_len :
        src_spl.append(src)
    else :
        src_bufs = token(src, ".\"")

        for _ in range(len(src_bufs)):
            src_cur = src_bufs.pop(0)

            if len(src_cur) < max_len :
                src_bufs.append(src_cur)
                #src_spl.append(src_cur)
            else : 
                src_bufs2 = token(src_cur, "──")
                for src_buf2 in src_bufs2:
                    src_bufs.append(src_buf2)
        
        for _ in range(len(src_bufs)):
            src_cur = src_bufs.pop(0)

            if len(src_cur) < max_len :
                src_bufs.append(src_cur)
                #src_spl.append(src_cur)
            else : 
                src_bufs2 = token(src_cur, ", ")
                for src_buf2 in src_bufs2:
                    src_bufs.append(src_buf2)

        for _ in range(len(src_bufs)):
            src_cur = src_bufs.pop(0)

            if len(src_cur) < max_len :
                src_bufs.append(src_cur)
                #src_spl.append(src_cur)
            else : 
                src_bufs2 = left_token(src_cur, "▲ ")
                for src_buf2 in src_bufs2:
                    src_bufs.append(src_buf2)

        for _ in range(len(src_bufs)):
            src_cur = src_bufs.pop(0)

            if len(src_cur) < max_len :
                src_bufs.append(src_cur)
                #src_spl.append(src_cur)
            else : 
                src_bufs2 = token(src_cur, "/ ")
                for src_buf2 in src_bufs2:
                    src_bufs.append(src_buf2)

        for _ in range(len(src_bufs)):
            src_cur = src_bufs.pop(0)

            if len(src_cur) < max_len :
                src_bufs.append(src_cur)
            else : 
                src_bufs2 = left_token(src_cur, "△ ")
                for src_buf2 in src_bufs2:
                    src_bufs.append(src_buf2)
        
        #last
        for _ in range(len(src_bufs)):
            src_cur = src_bufs.pop(0)

            if len(src_cur) < max_len :
                src_bufs.append(src_cur)
            else : 
                src_bufs2 = last(src_cur, org_max_len)
                for src_buf2 in src_bufs2:
                    src_bufs.append(src_buf2)
    
        for src_buf in src_bufs:
            assert len(src_buf) < max_len, f"wrong split"
            src_spl.append(src_buf)

    assert "".join(src_spl) == original_sent, "wrong split"

    return src_spl

