import os
import numpy as np
import evaluations

def evaluate(args, trainer, dataset, train_theta, test_theta, current_run_dir, logger, wandb = None, read_labels=True, eval_tag=None):
    metrics = {}
    if eval_tag:
        print(f"[EVAL] {eval_tag}")
        logger.info(f"[EVAL] {eval_tag}")
    top_words_10 = trainer.save_top_words(
        dataset.vocab, 10, current_run_dir)
    top_words_15 = trainer.save_top_words(
        dataset.vocab, 15, current_run_dir)
    top_words_20 = trainer.save_top_words(
        dataset.vocab, 20, current_run_dir)
    top_words_25 = trainer.save_top_words(
        dataset.vocab, 25, current_run_dir)

    # argmax of train and test theta
    # train_theta_argmax = train_theta.argmax(axis=1)
    # test_theta_argmax = test_theta.argmax(axis=1)
    train_theta_argmax = train_theta.argmax(axis=1)
    unique_elements, counts = np.unique(train_theta_argmax, return_counts=True)
    print(f'train theta argmax: {unique_elements, counts}')
    logger.info(f'train theta argmax: {unique_elements, counts}')
    test_theta_argmax = test_theta.argmax(axis=1)
    unique_elements, counts = np.unique(test_theta_argmax, return_counts=True)
    print(f'test theta argmax: {unique_elements, counts}')
    logger.info(f'test theta argmax: {unique_elements, counts}')

    TD_15 = evaluations.compute_topic_diversity(
        top_words_15, _type="TD")
    print(f"TD_15: {TD_15:.5f}")
    # wandb.log({"TD_15": TD_15})
    logger.info(f"TD_15: {TD_15:.5f}")
    metrics["TD_15"] = float(TD_15)

    # evaluating clustering
    if read_labels:
        clustering_results = evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])
        print(f"InversePurity: ", clustering_results['InversePurity'])
        print(f"HarmonicPurity: ", clustering_results['HarmonicPurity'])
        print(f"ARI: ", clustering_results['ARI'])
        # wandb.log({"NMI": clustering_results['NMI']})
        # wandb.log({"Purity": clustering_results['Purity']})
        logger.info(f"NMI: {clustering_results['NMI']}")
        logger.info(f"Purity: {clustering_results['Purity']}")
        logger.info(f"InversePurity: {clustering_results['InversePurity']}")
        logger.info(f"HarmonicPurity: {clustering_results['HarmonicPurity']}")
        logger.info(f"ARI: {clustering_results['ARI']}")
        metrics["NMI"] = float(clustering_results["NMI"])
        metrics["Purity"] = float(clustering_results["Purity"])
        metrics["InversePurity"] = float(clustering_results["InversePurity"])
        metrics["HarmonicPurity"] = float(clustering_results["HarmonicPurity"])
        metrics["ARI"] = float(clustering_results["ARI"])

    # evaluate classification
    if read_labels:
        classification_results = evaluations.evaluate_classification(
            train_theta, test_theta, dataset.train_labels, dataset.test_labels, tune=args.tune_SVM)
        print(f"Accuracy: ", classification_results['acc'])
        # wandb.log({"Accuracy": classification_results['acc']})
        logger.info(f"Accuracy: {classification_results['acc']}")
        print(f"Macro-f1", classification_results['macro-F1'])
        # wandb.log({"Macro-f1": classification_results['macro-F1']})
        logger.info(f"Macro-f1: {classification_results['macro-F1']}")
        metrics["Accuracy"] = float(classification_results["acc"])
        metrics["Macro-f1"] = float(classification_results["macro-F1"])

    # LLM eval
    if getattr(args, "enable_llm_eval", True):
        llm_scores, llm_mean = evaluations.llm_eval(
            top_words_15, llm_model=args.update_llm_model, out_dir=current_run_dir, resume=True
        )
        print(f"LLM_eval_mean: {llm_mean:.5f}")
        logger.info(f"LLM_eval_mean: {llm_mean:.5f}")
        logger.info(f"LLM_eval_scores: {llm_scores}")
        metrics["LLM_eval_mean"] = float(llm_mean)

    try:
        tc_wiki_scores, tc_wiki_mean = evaluations.topic_coherence.TC_on_wikipedia_llm_itl(
            os.path.join(current_run_dir, "top_words_15.txt"),
            tc_metric="C_V",
        )
        print(f"TC_wikipedia_llm_itl: {tc_wiki_mean:.5f}")
        logger.info(f"TC_wikipedia_llm_itl: {tc_wiki_mean:.5f}")
        logger.info(f"TC_wikipedia_llm_itl_scores: {tc_wiki_scores}")
        metrics["TC_wikipedia_llm_itl"] = float(tc_wiki_mean)
    except Exception as exc:
        print(f"TC_wikipedia_llm_itl failed: {exc}")
        logger.info(f"TC_wikipedia_llm_itl failed: {exc}")

    try:
        npmi_wiki_scores, npmi_wiki_mean = evaluations.topic_coherence.TC_on_wikipedia_llm_itl(
            os.path.join(current_run_dir, "top_words_15.txt"),
            tc_metric="NPMI",
        )
        print(f"NPMI_wiki: {npmi_wiki_mean:.5f}")
        logger.info(f"NPMI_wiki: {npmi_wiki_mean:.5f}")
        logger.info(f"NPMI_wiki_scores: {npmi_wiki_scores}")
        metrics["NPMI_wiki"] = float(npmi_wiki_mean)
    except Exception as exc:
        print(f"NPMI_wiki failed: {exc}")
        logger.info(f"NPMI_wiki failed: {exc}")

    # wandb.finish()
    return metrics
