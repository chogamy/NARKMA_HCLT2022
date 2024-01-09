def last(src, morph, tag, max_len=200):

    src_bufs = src.rsplit(" ")
    morph_bufs = morph.rsplit(" ")

    assert len(src_bufs) == len(morph_bufs), f"src, morph len different"

    # append deliimter
    for i, (src_buf, morph_buf) in enumerate(zip(src_bufs, morph_bufs)):
        if src_bufs[-1] != src_buf:
            src_bufs[i] += " "
            morph_bufs[i] += " "

    src_bufs2 = []
    morph_bufs2 = []

    src_result = []
    morph_result = []


    # splitted by space unit
    for src_buf, morph_buf in zip(src_bufs, morph_bufs):
        src_bufs2.append(src_buf)
        morph_bufs2.append(morph_buf)

        if len("".join(src_bufs2)) > (max_len / 2) or len("".join(morph_bufs2)) > (max_len / 2):
            src_result.append("".join(src_bufs2))
            morph_result.append("".join(morph_bufs2))

            src_bufs2 = []
            morph_bufs2 = []
    src_result.append("".join(src_bufs2))
    morph_result.append("".join(morph_bufs2))

    src_bufs = src_result
    morph_bufs = morph_result

    if src_bufs[-1] == "" and morph_bufs[-1] == "":
        src_bufs.pop()
        morph_bufs.pop()

    assert "".join(src_bufs) == src, f"src not splitted"
    assert "".join(morph_bufs) == morph, f"morph not splitted"
            

    # append tag
    tag_bufs = []
    tag_origin = tag
    tag = tag.split(" ")
    for morph_buf in morph_bufs:
        tag_buf = []
        for i in range(len(morph_buf)):
            tag_buf.append(tag.pop(0))
        tag_buf = " ".join(tag_buf)
        tag_bufs.append(tag_buf)

    assert " ".join(tag_bufs) == tag_origin, f"wrong tag"
        
                
    return src_bufs, morph_bufs, tag_bufs


def left_token(src, morph, tag, src_token, morph_token, max_len=200):

    src_bufs = src.rsplit(src_token)
    morph_bufs = morph.rsplit(morph_token)

    # append delimiter
    for i, (src_buf, morph_buf) in enumerate(zip(src_bufs, morph_bufs)):
        if src_bufs[0] != src_buf:
            src_bufs[i] = src_token + src_bufs[i]
            morph_bufs[i] = morph_token + morph_bufs[i]

    # when first is empty
    if src_bufs[0] == "" and morph_bufs[0] == "":
        src_bufs.pop(0)
        morph_bufs.pop(0)

    assert "".join(src_bufs) == src, f"src not splitted"
    assert "".join(morph_bufs) == morph, f"morph not splitted"
            

    # append tag
    tag_bufs = []
    tag_origin = tag
    tag = tag.split(" ")
    for morph_buf in morph_bufs:
        tag_buf = []
        for i in range(len(morph_buf)):
            tag_buf.append(tag.pop(0))
        tag_buf = " ".join(tag_buf) 
        tag_bufs.append(tag_buf)

    assert " ".join(tag_bufs) == tag_origin, f"wrong tag"
                
    return src_bufs, morph_bufs, tag_bufs


def token(src, morph, tag, src_token, morph_token, max_len=200):

    src_bufs = src.rsplit(src_token)
    morph_bufs = morph.rsplit(morph_token)

    assert len(src_bufs) == len(morph_bufs), f"src, morph len different"

    # append delimter
    for i, (src_buf, morph_buf) in enumerate(zip(src_bufs, morph_bufs)):
        if src_bufs[-1] != src_buf:
            src_bufs[i] += src_token
            morph_bufs[i] += morph_token        
            
    assert "".join(src_bufs) == src, f"src not splitted"
    assert "".join(morph_bufs) == morph, f"morph not splitted"
            

    # append tag
    tag_bufs = []
    tag_origin = tag
    tag = tag.split(" ")
    for morph_buf in morph_bufs:
        tag_buf = []
        for i in range(len(morph_buf)):
            tag_buf.append(tag.pop(0))
        tag_buf = " ".join(tag_buf)
        tag_bufs.append(tag_buf)

    assert " ".join(tag_bufs) == tag_origin, f"wrong tag"
                
    return src_bufs, morph_bufs, tag_bufs


def split(src : str, morph : str, tag : str, max_len = 200):

    src_spl = []
    morph_spl = []
    tag_spl = []

    if len(src) < max_len and len(morph) < max_len :
        src_spl.append(src)
        morph_spl.append(morph)
        tag_spl.append(tag)
    else :
        src_bufs, morph_bufs, tag_bufs = token(src, morph, tag, src_token=".\"", morph_token=".+\"", max_len=max_len)

        for _ in range(len(src_bufs)):
            src_cur = src_bufs.pop(0)
            morph_cur = morph_bufs.pop(0)
            tag_cur = tag_bufs.pop(0)

            if len(src_cur) < max_len and len(morph_cur) < max_len:
                src_bufs.append(src_cur)
                morph_bufs.append(morph_cur)
                tag_bufs.append(tag_cur)
            else :
                src_bufs2, morph_bufs2, tag_bufs2 = token(src_cur, morph_cur, tag_cur, "──", "─+─", max_len=max_len)
                for src_buf2, morph_buf2, tag_buf2 in zip(src_bufs2, morph_bufs2, tag_bufs2):
                    src_bufs.append(src_buf2)
                    morph_bufs.append(morph_buf2)
                    tag_bufs.append(tag_buf2)

        for _ in range(len(src_bufs)):
            src_cur = src_bufs.pop(0)
            morph_cur = morph_bufs.pop(0)
            tag_cur = tag_bufs.pop(0)

            if len(src_cur) < max_len and len(morph_cur) < max_len:
                src_bufs.append(src_cur)
                morph_bufs.append(morph_cur)
                tag_bufs.append(tag_cur)
            else :
                src_bufs2, morph_bufs2, tag_bufs2 = token(src_cur, morph_cur, tag_cur, src_token=", ", morph_token=", ", max_len=max_len)
                for src_buf2, morph_buf2, tag_buf2 in zip(src_bufs2, morph_bufs2, tag_bufs2):
                    src_bufs.append(src_buf2)
                    morph_bufs.append(morph_buf2)
                    tag_bufs.append(tag_buf2)
        
        for _ in range(len(src_bufs)):
            src_cur = src_bufs.pop(0)
            morph_cur = morph_bufs.pop(0)
            tag_cur = tag_bufs.pop(0)

            if len(src_cur) < max_len and len(morph_cur) < max_len:
                src_bufs.append(src_cur)
                morph_bufs.append(morph_cur)
                tag_bufs.append(tag_cur)
            else :
                src_bufs2, morph_bufs2, tag_bufs2 = left_token(src_cur, morph_cur, tag_cur, "▲ ", "▲ ", max_len=max_len)
                for src_buf2, morph_buf2, tag_buf2 in zip(src_bufs2, morph_bufs2, tag_bufs2):
                    src_bufs.append(src_buf2)
                    morph_bufs.append(morph_buf2)
                    tag_bufs.append(tag_buf2)
        
        for _ in range(len(src_bufs)):
            src_cur = src_bufs.pop(0)
            morph_cur = morph_bufs.pop(0)
            tag_cur = tag_bufs.pop(0)

            if len(src_cur) < max_len and len(morph_cur) < max_len:
                src_bufs.append(src_cur)
                morph_bufs.append(morph_cur)
                tag_bufs.append(tag_cur)
            else :
                src_bufs2, morph_bufs2, tag_bufs2 = token(src_cur, morph_cur, tag_cur, "/ ", "/ ", max_len=max_len)
                for src_buf2, morph_buf2, tag_buf2 in zip(src_bufs2, morph_bufs2, tag_bufs2):
                    src_bufs.append(src_buf2)
                    morph_bufs.append(morph_buf2)
                    tag_bufs.append(tag_buf2)
        
        for _ in range(len(src_bufs)):
            src_cur = src_bufs.pop(0)
            morph_cur = morph_bufs.pop(0)
            tag_cur = tag_bufs.pop(0)

            if len(src_cur) < max_len and len(morph_cur) < max_len:
                src_bufs.append(src_cur)
                morph_bufs.append(morph_cur)
                tag_bufs.append(tag_cur)
            else :
                src_bufs2, morph_bufs2, tag_bufs2 = left_token(src_cur, morph_cur, tag_cur, "△ ", "△ ", max_len=max_len)
                for src_buf2, morph_buf2, tag_buf2 in zip(src_bufs2, morph_bufs2, tag_bufs2):
                    src_bufs.append(src_buf2)
                    morph_bufs.append(morph_buf2)
                    tag_bufs.append(tag_buf2)

        #last
        for _ in range(len(src_bufs)):
            src_cur = src_bufs.pop(0)
            morph_cur = morph_bufs.pop(0)
            tag_cur = tag_bufs.pop(0)

            if len(src_cur) < max_len and len(morph_cur) < max_len:
                src_bufs.append(src_cur)
                morph_bufs.append(morph_cur)
                tag_bufs.append(tag_cur)
            else :
                src_bufs2, morph_bufs2, tag_bufs2 = last(src_cur, morph_cur, tag_cur, max_len=max_len)
                for src_buf2, morph_buf2, tag_buf2 in zip(src_bufs2, morph_bufs2, tag_bufs2):
                    src_bufs.append(src_buf2)
                    morph_bufs.append(morph_buf2)
                    tag_bufs.append(tag_buf2)

        
        for src_buf, morph_buf, tag_buf in zip(src_bufs, morph_bufs, tag_bufs):
            src_spl.append(src_buf)
            morph_spl.append(morph_buf)
            tag_spl.append(tag_buf)
            assert len(src_buf) < max_len and len(morph_buf) < max_len, f"not splitted"

    return src_spl, morph_spl, tag_spl