
CUDA_VISIBLE_DEVICES=1,2 python finetune/run_classifier_cv.py  --pretrained_model_path /home/tyx/wch/pretrained_models/mixed_corpus_roberta_wwm_xlarge_model.bin \
                                                             --vocab_path models/google_zh_vocab.txt \
                                                             --output_model_path models/test_ecisa/ecisa_classifier_model_0.bin \
                                                             --config_path models/bert_xlarge_config.json \
                                                             --train_path datasets/smp2019-ecisa/train.tsv \
                                                             --train_features_path datasets/test_ecisa/train_features_0.npy \
                                                             --pooling mean \
                                                             --folds_num 10 --epochs_num 3 --batch_size 1 --accumulation_steps 32 \
                                                             --embedding word_pos_seg --encoder transformer --mask fully_visible &> train_model_0.log


CUDA_VISIBLE_DEVICES=1,2 python finetune/run_classifier_cv.py  --pretrained_model_path /home/tyx/wch/pretrained_models/mixed_corpus_roberta_wwm_xlarge_model.bin \
                                                             --vocab_path models/google_zh_vocab.txt \
                                                             --output_model_path models/test_ecisa/ecisa_classifier_model_1.bin \
                                                             --config_path models/bert_xlarge_config.json \
                                                             --train_path datasets/smp2019-ecisa/train.tsv \
                                                             --train_features_path datasets/test_ecisa/train_features_1.npy \
                                                             --pooling max \
                                                             --folds_num 10 --epochs_num 3 --batch_size 1 --accumulation_steps 32 \
                                                             --embedding word_pos_seg --encoder transformer --mask fully_visible &> train_model_1.log


CUDA_VISIBLE_DEVICES=1,2 python finetune/run_classifier_cv.py  --pretrained_model_path /home/tyx/wch/pretrained_models/mixed_corpus_roberta_wwm_xlarge_model.bin \
                                                             --vocab_path models/google_zh_vocab.txt \
                                                             --output_model_path models/test_ecisa/ecisa_classifier_model_2.bin \
                                                             --config_path models/bert_xlarge_config.json \
                                                             --train_path datasets/smp2019-ecisa/train.tsv \
                                                             --train_features_path datasets/test_ecisa/train_features_2.npy \
                                                             --folds_num 10 --epochs_num 3 --batch_size 1 --accumulation_steps 32 \
                                                             --embedding word_pos_seg --encoder transformer --mask fully_visible &> train_model_2.log

CUDA_VISIBLE_DEVICES=1,2 python finetune/run_classifier_cv.py  --pretrained_model_path /home/tyx/wch/pretrained_models/mixed_corpus_roberta_wwm_xlarge_model.bin \
                                                             --vocab_path models/google_zh_vocab.txt \
                                                             --output_model_path models/test_ecisa/ecisa_classifier_model_3.bin \
                                                             --config_path models/bert_xlarge_config.json \
                                                             --train_path datasets/smp2019-ecisa/train.tsv \
                                                             --train_features_path datasets/test_ecisa/train_features_3.npy \
                                                             --pooling last \
                                                             --folds_num 10 --epochs_num 3 --batch_size 1 --accumulation_steps 32 \
                                                             --embedding word_pos_seg --encoder transformer --mask fully_visible &> train_model_3.log
# # done
# CUDA_VISIBLE_DEVICES=1,2 python finetune/run_classifier_cv.py --pretrained_model_path /home/tyx/chwu/pretrained_models/xlm_roberta_base_pytorch.bin \
#                                                                --spm_model_path models/xlmroberta_spm.model \
#                                                                --output_model_path models/test_ecisa/ecisa_classifier_model_3.bin \
#                                                                --config_path models/xlm-roberta/base_config.json \
#                                                                --train_path datasets/smp2019-ecisa/train.tsv \
#                                                                --train_features_path datasets/test_ecisa/train_features_3.npy \
#                                                                --folds_num 5 --epochs_num 3 --batch_size 8 \
#                                                                --embedding word_pos_seg --encoder transformer --mask fully_visible &> train_model_3.log

CUDA_VISIBLE_DEVICES=1,2 python finetune/run_classifier_cv.py  --pretrained_model_path /home/tyx/wch/pretrained_models/mixed_corpus_bert_xlarge_model.bin \
                                                             --vocab_path models/google_zh_vocab.txt \
                                                             --output_model_path models/test_ecisa/ecisa_classifier_model_4.bin \
                                                             --config_path models/bert_xlarge_config.json \
                                                             --train_path datasets/smp2019-ecisa/train.tsv \
                                                             --train_features_path datasets/test_ecisa/train_features_4.npy \
                                                             --folds_num 10 --epochs_num 3 --batch_size 1 --accumulation_steps 32 \
                                                             --embedding word_pos_seg --encoder transformer --mask fully_visible  &> train_model_4.log
# discarded
# CUDA_VISIBLE_DEVICES=7 python finetune/run_classifier_cv.py --pretrained_model_path /home/chwu/pretrained_models/chinese_xlnet_mid_uer_pytorch.bin \
#                                                                --vocab_path models/google_zh_vocab.txt \
#                                                                --output_model_path models/ecisa_classifier_model_1.bin \
#                                                                --config_path models/xlnet/mid_config.json \
#                                                                --train_path datasets/smp2019-ecisa/train.tsv \
#                                                                --train_features_path datasets/smp2019-ecisa/train_features_1.npy \
#                                                                --folds_num 5 --epochs_num 3 --batch_size 8 --accumulation_steps 6 \
#                                                                --embedding word_pos_seg --encoder transformer --mask fully_visible
