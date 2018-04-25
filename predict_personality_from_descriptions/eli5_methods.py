
class StaticClass(object):

# Klass.static_method()
    @staticmethod
    def _logit_model_eli5_explain_weights(model_type, cur_key, cur_trait, score, clf, vectorizer, eli_5_dir, logging, cur_time):
            """Explains top K features with the highest coefficient values, per class, using eli5"""

            import os
            from eli5 import sklearn
            from eli5 import formatters

            logging.info("_base_logit_model_eli5_explain_weights trait = {}, score = {}, model_type = {}, parameters = {}".format(cur_trait, score, model_type, cur_key))

            # get explanation
            eli5_ew = sklearn.explain_linear_classifier_weights(
                clf=clf,
                vec=vectorizer,
                top=40,
                target_names=['under', 'over']
            )
            # format explanation as html
            eli5_fh = formatters.format_as_html(eli5_ew)

            # create relevant sub dir for output files
            # output_path_for_vertical = self.params['eli5_output_path_eli5'] + '/' + vertical + '/weights'
            # output_path_for_vertical = log + '/' + vertical + '/weights'
            output_path = eli_5_dir + '/' + str(cur_time) + '/weights'
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # write explanation html to file
            wf_path = output_path + '/{}_{}_{}_{}.html' \
                .format(round(score, 3), cur_trait, model_type, cur_key)

            prefix_to_html = ('Personality trait: ' + str(cur_trait) + ', Score: ' + str(score) +
                              '<br><br>' + 'Model: ' + str(model_type) + '<br><br>' + 'Parameters: ' +
                              str(cur_key) + '<br><br>').encode('utf-8', 'replace')

            lines_final = prefix_to_html + eli5_fh.encode('utf8', 'replace')

            logging.info("writing weight explanation to file {}".format(wf_path))
            with open(wf_path, 'w') as wf:
                wf.writelines(lines_final)

    @staticmethod
    # (model_type, cur_key, cur_trait, score, clf, vectorizer, eli_5_dir, logging, cur_time):
    def _logit_model_eli5_explain_prediction(model_type, cur_key, cur_trait, score, auc_score, clf, vectorizer, X_raw,
                                             X_vectorized, Y, eli_5_dir, logging, cur_time, k=10):
            """Explains top K predictions with the highest confidence where the model was correct, per class, using eli5"""

            import copy
            import numpy as np
            from eli5 import sklearn
            from eli5 import formatters
            import os

            logging.info("_base_logit_model_eli5_explain_prediction trait = {}, score = {}, model_type = {}, parameters = {}".format(
                    cur_trait, score, model_type, cur_key))

            num_labels = len(np.unique(Y))
            assert num_labels == 2, "currently supporting evaluation for binary problems only"

            '''assert 'description' in self.item_data_logit_features_list

            # get optimal threshold for stest
            sdt = 'stest'
            assert 'threshold_value' in self.item_data_logit_models[model_type][vertical] \
                [price_col, category_col][sdt]['threshold_opt']'''

            if False:
                # TODO
                threshold_opt = 'calculate' # self.item_data_logit_models[model_type][vertical][price_col, category_col][sdt]['threshold_opt']['threshold_value']
            else:
                threshold_opt = 0.5

            # get probability predictions for input data
            Y_probs = clf.predict_proba(X_vectorized)

            # get binary scores from probabilities using threshold
            Y_pred = (Y_probs[:, 1] > threshold_opt).astype(np.int)
            Y_wrong = np.logical_not(np.equal(Y, Y_pred))
            # Y_wrong = np.logical_not(np.equal(Y, Y_probs))

            # place threshold value for all wrong locations and by doing that
            # make sure they will not be picked as most extreme when sorting
            Y_probs_adjusted = copy.deepcopy(Y_probs[:, 1])
            Y_probs_adjusted[Y_wrong] = threshold_opt

            # take top k samples where the model had the greatest confidence and it was correct
            np_topk = np.argsort(Y_probs_adjusted)[-k:]
            np_bottomk = np.argsort(Y_probs_adjusted)[:k]

            dict_of_lists = {'over': np_topk, 'under': np_bottomk}

            for position, np_list in dict_of_lists.iteritems():

                for i, loc in enumerate(np_list):

                    # raw = X_raw.iloc[[loc]]['item_description'].values[0]
                    raw = X_raw[loc]

                    # print(raw)
                    # print("real: {}, model probability: {}".format(Y.iloc[loc], Y_probs[loc, 1]))

                    prefix_to_html = ''
                    try:
                        a = 4
                        # real_pred = "true label: {}, model probability: {} (model prediction: {})<br><br>" \
                        #    .format(Y.iloc[loc], Y_probs[loc, 1], Y_pred[loc])
                        '''real_pred = "true label: {}, model probability: {} (model prediction: {})<br><br>" \
                            .format(Y.iloc[loc], Y_probs[loc, 1], Y_probs[loc])

                        logging.info(
                            "_base_logit_model_eli5_explain_prediction = {}, score = {}, model_type = {}, parameters = {}".format(
                                cur_trait, score, model_type, cur_key))

                        prefix_to_html = ('Personality trait: ' + str(cur_trait) + '<br><br>' + 'parameters: ' +
                                          str(cur_key) + '<br><br>' + real_pred + 'item_description:<br>' + raw).encode('utf-8', 'replace')'''

                    except UnicodeDecodeError:
                        pass
                        # TODO: handle this properly

                    # get explanation
                    eli5_ep = sklearn.explain_prediction_linear_classifier(clf=clf, doc=X_vectorized[loc],
                                                                           vec=vectorizer, top=20,
                                                                           target_names=['under', 'over'], vectorized=True)

                    # format explanation as html
                    eli5_fh = formatters.format_as_html(explanation=eli5_ep)

                    if position == 'under':
                        # manually replace red and green within the html
                        symbol_temp = "tempTEMPtempTEMPtemp"
                        eli5_fh = eli5_fh.replace("hsl(0", symbol_temp)
                        eli5_fh = eli5_fh.replace("hsl(120", "hsl(0")
                        eli5_fh = eli5_fh.replace(symbol_temp, "hsl(120")

                    '''
                    output_path = eli_5_dir + '/' + str(cur_time)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    # write explanation html to file
                    wf_path = output_path + '/{}_{}_{}_{}.html' \
                        .format(round(score, 3), cur_trait, model_type, cur_key)

                    prefix_to_html = ('Personality trait: ' + str(cur_trait) + ', Score: ' + str(score) +
                                      '<br><br>' + 'Model: ' + str(model_type) + '<br><br>' + 'Parameters: ' +
                                      str(cur_key) + '<br><br>').encode('utf-8', 'replace')
                                      '''

                    # create relevant sub dir for output files
                    output_path = eli_5_dir + '/' + str(cur_time) + '/predictions/' + str(round(score, 3)) + \
                                  '_accuracy' + '_auc_' + str(round(auc_score, 3)) + '_' + cur_trait + '_' + model_type \
                                  + '_' + cur_key
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    # write explanation html to file
                    wf_path = output_path + '/type_{}_loc_{}_pred_{}.html' \
                        .format(position, loc, round(Y_probs_adjusted[loc], 4))

                    lines_final = prefix_to_html + eli5_fh.encode('utf8', 'replace')

                    # logging.info("writing prediction explanation to file".format(wf_path))

                    with open(wf_path, 'w') as wf:
                        wf.write(lines_final)
