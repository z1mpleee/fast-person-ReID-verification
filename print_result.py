def print_result(all_cmc, mAP, ranks):
    pr = ""
    for r in ranks:
        pr += ", Rank-{}: {:.2%}".format(r, all_cmc[r-1])
    print("mAP: {:.2%}".format(mAP) + pr)
