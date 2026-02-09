import os
import numpy as np
import evaluations

def evaluate(args, trainer, dataset, train_theta, test_theta, current_run_dir, logger, wandb = None, read_labels=True):
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

    # TD_15 = evaluations.compute_topic_diversity(
    #     top_words_15, _type="TD")
    # print(f"TD_15: {TD_15:.5f}")


    # # evaluating clustering
    # if read_labels:
    #     clustering_results = evaluations.evaluate_clustering(
    #         test_theta, dataset.test_labels)
    #     print(f"NMI: ", clustering_results['NMI'])
    #     print(f'Purity: ', clustering_results['Purity'])


    # TC_15_list, TC_15 = evaluations.topic_coherence.TC_on_wikipedia(
    #     os.path.join(current_run_dir, 'top_words_15.txt'))
    # print(f"TC_15: {TC_15:.5f}")
    # TD_10 = evaluations.compute_topic_diversity(
    #     top_words_10, _type="TD")
    # print(f"TD_10: {TD_10:.5f}")
    # wandb.log({"TD_10": TD_10})
    # logger.info(f"TD_10: {TD_10:.5f}")

    TD_15 = evaluations.compute_topic_diversity(
        top_words_15, _type="TD")
    print(f"TD_15: {TD_15:.5f}")
    # wandb.log({"TD_15": TD_15})
    logger.info(f"TD_15: {TD_15:.5f}")
    
    # IRBO
    # IRBO_15 = evaluations.buubyyboo_dth([top_words.split() for top_words in top_words_15], topk=15)
    # print(f"IRBO_15: {IRBO_15:.5f}")
    # # wandb.log({"IRBO_15": IRBO_15})
    # logger.info(f"IRBO_15: {IRBO_15:.5f}")

    # TD_20 = topmost.evaluations.compute_topic_diversity(
    #     top_words_20, _type="TD")
    # print(f"TD_20: {TD_20:.5f}")
    # wandb.log({"TD_20": TD_20})
    # logger.info(f"TD_20: {TD_20:.5f}")

    # TD_25 = topmost.evaluations.compute_topic_diversity(
    #     top_words_25, _type="TD")
    # print(f"TD_25: {TD_25:.5f}")
    # wandb.log({"TD_25": TD_25})
    # logger.info(f"TD_25: {TD_25:.5f}")

    # evaluating clustering
    if read_labels:
        clustering_results = evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])
        # wandb.log({"NMI": clustering_results['NMI']})
        # wandb.log({"Purity": clustering_results['Purity']})
        logger.info(f"NMI: {clustering_results['NMI']}")
        logger.info(f"Purity: {clustering_results['Purity']}")

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

    # TC on train texts
    TC_train_list, TC_train = evaluations.compute_topic_coherence(
        dataset.train_texts, dataset.vocab, top_words_15, cv_type='c_v')
    print(f"TC_train: {TC_train:.5f}")
    # wandb.log({"TC_train": TC_train})
    logger.info(f"TC_train: {TC_train:.5f}")
    logger.info(f'TC_train list: {TC_train_list}')

    # wandb.finish()