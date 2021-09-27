CUDA_VISIBLE_DEVICES=1 python inference/run_classifier_infer_cv.py --load_model_path models/test_ecisa/ecisa_classifier_model_0.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert_xlarge_config.json \
                                                                    --test_path datasets/smp2019-ecisa/dev.tsv \
                                                                    --test_features_path datasets/smp2019-ecisa/test_features_0.npy \
                                                                    --folds_num 10 --labels_num 3 \
                                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible
                                                                    
CUDA_VISIBLE_DEVICES=1 python inference/run_classifier_infer_cv.py --load_model_path models/test_ecisa/ecisa_classifier_model_1.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert_xlarge_config.json \
                                                                    --test_path datasets/smp2019-ecisa/dev.tsv \
                                                                    --test_features_path datasets/smp2019-ecisa/test_features_1.npy \
                                                                    --folds_num 10 --labels_num 3 \
                                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=1 python inference/run_classifier_infer_cv.py --load_model_path models/test_ecisa/ecisa_classifier_model_2.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert_xlarge_config.json \
                                                                    --test_path datasets/smp2019-ecisa/dev.tsv \
                                                                    --test_features_path datasets/smp2019-ecisa/test_features_2.npy \
                                                                    --folds_num 10 --labels_num 3 \
                                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=1 python inference/run_classifier_infer_cv.py --load_model_path models/test_ecisa/ecisa_classifier_model_3.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert_xlarge_config.json \
                                                                    --test_path datasets/smp2019-ecisa/dev.tsv \
                                                                    --test_features_path datasets/smp2019-ecisa/test_features_3.npy \
                                                                    --folds_num 10 --labels_num 3 \
                                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=1 python inference/run_classifier_infer_cv.py --load_model_path models/test_ecisa/ecisa_classifier_model_4.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert_xlarge_config.json \
                                                                    --test_path datasets/smp2019-ecisa/dev.tsv \
                                                                    --test_features_path datasets/smp2019-ecisa/test_features_4.npy \
                                                                    --folds_num 10 --labels_num 3 \
                                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible