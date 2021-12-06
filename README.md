# fast-person-ReID-verification base on [fast-reid](https://github.com/JDAI-CV/fast-reid)

The purpose of this project is to accelerate the speed of ReID verification and integrate the re-ranking code

## Compile with cython to accelerate evalution
    $ git clone https://github.com/z1mpleee/fast-person-ReID-verification rank_cylib
    $ cd rank_cylib
    $ make all

## quick start guide
    from rank_cylib import evaluate_rank

    cmc, mAP, _ = evaluate_rank(qf, gf, lqf, lgf, q_pids, g_pids, q_camids, g_camids, aligned=aligned, reranking=True, use_cython=True)

## thanks to [fast-reid](https://github.com/JDAI-CV/fast-reid)
