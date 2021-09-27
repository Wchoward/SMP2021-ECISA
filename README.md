1. 数据预处理

   ```shell
   cd data_preprocess
   python clean_data.py
   ```

   生成模型输入的`.tsv` 文件与数据清洗过程后出现为空数据的数据id文件`neural_idx_in_test.txt`

   

2. 训练

   2.1	首先使用10折交叉验证训练，得到分类器和训练集特征：

   ```shell
   cd UER-py
   sh train.sh
   ```

   训练过程中采用的预训练模型有：

   - mixed_corpus_roberta_wwm_xlarge_model.bin （https://share.weiyun.com/UsI0OSeR）
   - mixed_corpus_bert_xlarge_model.bin （https://share.weiyun.com/J9rj9WRB）

   `train.sh`文件中 `pretrained_model_path`为预训练模型的路径

   2.2	使用K折交叉验证推理，得到验证集特征：

   ```shell
   sh eval.sh
   ```

   2.3	使用LightGBM超参数搜索：

   ```shell
   python scripts/run_lgb_cv_bayesopt.py --train_path datasets/smp2019-ecisa/train.tsv \
                                          --train_features_path datasets/test_ecisa/ \
                                          --models_num 5 --folds_num 10 --labels_num 3 --epochs_num 100
   ```

   2.4	使用LightGBM进行训练和验证，并保存得到集成模型`stack_model.txt`：

   ```shell
   python scripts/run_lgb.py --train_path datasets/smp2019-ecisa/train.tsv --test_path datasets/smp2019-ecisa/dev.tsv \
                              --train_features_path datasets/test_ecisa/ --test_features_path datasets/smp2019-ecisa/ \
                              --models_num 5 --labels_num 3
   ```

   

3. 预测

   3.1	使用K折交叉验证推理，得到测试集特征：

   ```shell
   sh test.sh
   ```

   3.2	将测试集特征输入至训练好的集成模型中预测初步结果`result.txt`：

   ```shell
   python scripts/run_lgb_test.py --test_path datasets/smp2019-ecisa/test.tsv \
   --test_features_path datasets/smp2019-ecisa-result/ \
   --models_num 5 --labels_num 3
   ```

   3.3	根据数据清洗过程中得到的空数据的id更新测试集中的中性结果，得到测试集最终结果`final_result.txt`:

   ```shell
   python nerual.py
   ```

   

