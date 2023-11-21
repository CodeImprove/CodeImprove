1. First go to Defect-Prediction/code directory
2. Select the desired model_type, tokenizer_name=roberta-base, model_name_or_path for train.sh, test.sh, and emsemble.sh
3. To finetune with the desired model and dataset, run train.sh
4. To test the desired model and dataset, run.test.sh
5. For evaluation run, 
python ../evaluator/evaluator.py -a ../dataset/test.jsonl -p saved_models/predictions.txt 
to get the evaluation results.

6. For input validation, run emsemble.sh
Next run 
python3 ../evaluator/classify.py -a ../dataset/test.jsonl -p saved_models/predictions.txt -u saved_models/changed_uncertainty.txt to validate the inputs


8. Next go to ../ProgramTransformation folder and run transform.sh file for input adaptation phase.
9. Next, you can repeat step 4, and 5 to evaluate the model effectiveness. 



