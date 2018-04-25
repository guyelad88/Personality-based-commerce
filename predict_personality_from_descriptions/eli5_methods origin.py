def _logit_model_eli5_explain_weights(self, model_type, vertical, category_col, price_col, clf, vectorizer):
        """Explains top K features with the highest coefficient values, per class, using eli5"""

        Logger.info("_base_logit_model_eli5_explain_weights model_type = {}, vertical = {}"
                    .format(model_type, vertical))

        # get explanation
        eli5_ew = sklearn.explain_linear_classifier_weights(clf=clf, vec=vectorizer,
                                                            top=40, target_names=['under', 'over'])
        # format explanation as html
        eli5_fh = formatters.format_as_html(eli5_ew)

        # create relevant sub dir for output files
        output_path_for_vertical = self.params['eli5_output_path_eli5'] + '/' + vertical + '/weights'
        if not os.path.exists(output_path_for_vertical):
            os.makedirs(output_path_for_vertical)

        # write explanation html to file
        wf_path = output_path_for_vertical + '/eli5_weights_{}_{}_{}.html' \
            .format(model_type, category_col, price_col)

        prefix_to_html = ('vertical: ' + str(vertical) + '<br><br>' + 'category column: ' +
                          str(category_col) + '<br><br>').encode('utf-8', 'replace')

        lines_final = prefix_to_html + eli5_fh.encode('utf8', 'replace')

        Logger.debug("writing weight explanation to file {}".format(wf_path))
        with open(wf_path, 'w') as wf:
            wf.writelines(lines_final)

    def _logit_model_eli5_explain_prediction(self, model_type, vertical, category_col, price_col,
                                             clf, vectorizer, X_raw, X_vectorized, Y, k=10):
        """Explains top K predictions with the highest confidence where the model was correct, per class, using eli5"""

        Logger.info("_base_logit_model_eli5_explain_prediction model_type = {}, vertical = {}"
                    .format(model_type, vertical))

        num_labels = len(np.unique(Y))
        assert num_labels == 2, "currently supporting evaluation for binary problems only"

        assert 'description' in self.item_data_logit_features_list

        # get optimal threshold for stest
        sdt = 'stest'
        assert 'threshold_value' in self.item_data_logit_models[model_type][vertical] \
            [price_col, category_col][sdt]['threshold_opt']

        threshold_opt = self.item_data_logit_models[model_type][vertical] \
            [price_col, category_col][sdt]['threshold_opt']['threshold_value']

        # get probability predictions for input data
        Y_probs = clf.predict_proba(X_vectorized)

        # get binary scores from probabilities using threshold
        Y_pred = (Y_probs[:, 1] > threshold_opt).astype(np.int)

        Y_wrong = np.logical_not(np.equal(Y.values, Y_pred))

        # place threshold value for all wrong locations and by doing that
        # make sure they will not be picked as mot extreme when sorting
        Y_probs_adjusted = copy.deepcopy(Y_probs[:, 1])
        Y_probs_adjusted[Y_wrong] = threshold_opt

        # take top k samples where the model had the greatest confidence and it was correct
        np_topk = np.argsort(Y_probs_adjusted)[-k:]
        np_bottomk = np.argsort(Y_probs_adjusted)[:k]

        dict_of_lists = {'over': np_topk, 'under': np_bottomk}

        for position, np_list in dict_of_lists.iteritems():

            for i, loc in enumerate(np_list):

                raw = X_raw.iloc[[loc]]['item_description'].values[0]

                Logger.debug("----")
                Logger.debug(raw)
                Logger.debug("real: {}, model probability: {}".format(Y.iloc[loc], Y_probs[loc, 1]))

                prefix_to_html = ''
                try:
                    real_pred = "true label: {}, model probability: {} (model prediction: {})<br><br>" \
                        .format(Y.iloc[loc], Y_probs[loc, 1], Y_pred[loc])

                    prefix_to_html = ('vertical: ' + str(vertical) + '<br><br>' + 'category column: ' +
                                      str(category_col) + '<br><br>' + real_pred + 'item_description:<br>' + raw).encode('utf-8', 'replace')

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

                # create relevant sub dir for output files
                output_path_for_vertical = self.params['eli5_output_path_eli5'] + '/' + vertical + '/predictions'
                if not os.path.exists(output_path_for_vertical):
                    os.makedirs(output_path_for_vertical)

                # write explanation html to file
                wf_path = output_path_for_vertical + '/eli5_prediction_{}_{}_{}_{}_{}.html' \
                    .format(model_type, category_col, price_col, position, loc)

                lines_final = prefix_to_html + eli5_fh.encode('utf8', 'replace')

                Logger.debug("writing prediction explanation to file".format(wf_path))

                with open(wf_path, 'w') as wf:
                    wf.write(lines_final)
