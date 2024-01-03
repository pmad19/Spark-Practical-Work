#!/bin/bash

. ./run-env.sh

dataset=$(grep "^spark\.data\.default_input_file=" "$SPARK_PROPERTIES" | awk -F '=' '{print $2}')
exploratory_analysis=$(grep "^spark\.pipeline\.default_active_exploratory_analysis=" "$SPARK_PROPERTIES" | awk -F '=' '{print $2}')
preprocess=$(grep "^spark\.pipeline\.default_active_preprocess_job=" "$SPARK_PROPERTIES" | awk -F '=' '{print $2}')
fpr_fss=$(grep "^spark\.pipeline\.default_active_fpr_fss=" "$SPARK_PROPERTIES" | awk -F '=' '{print $2}')
fdr_fss=$(grep "^spark\.pipeline\.default_active_fdr_fss=" "$SPARK_PROPERTIES" | awk -F '=' '{print $2}')
fwe_fss=$(grep "^spark\.pipeline\.default_active_fwe_fss=" "$SPARK_PROPERTIES" | awk -F '=' '{print $2}')
train=$(grep "^spark\.pipeline\.default_active_train_job=" "$SPARK_PROPERTIES" | awk -F '=' '{print $2}')

echo "                                  _             ____  _  ___      "
echo " _ __ _    _ ___ _ __   __ _ _ __| | __        / ___|| ||__ \     "
echo "| '_ \ \  / / __| '_ \ / _\` | '__| |/ /  ___  | |  _ | |  ) |   "
echo "| |_) \ \/ /\__ \ |_) | (_| | |  |   <  |___| | |_| || | / /    "
echo "| .__/ \  / |___/ .__/ \__,_|_|  |_|\_\        \____||_||____|  "
echo "|_|    /_/      |_|                                             "
echo "=================================================================="


while [ $# -gt 0 ]; do
    key="$1"

    case $key in
        -d)
            dataset="$2"
            shift
            shift
            ;;
        -p)
            preprocess=true
            shift
            ;;
        -t)
            train=true
            shift
            ;;
        -e)
            exploratory_analysis=true
            shift
            ;;
        -fdr)
            fdr_fss=true
            shift
            ;;
        -fpr)
            fpr_fss=true
            shift
            ;;

        -fwe)
            fwe_fss=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ ! "$dataset" = "all" ]; then
  if [ ! "${dataset##*.}" = "csv" ]; then
    echo "The input -d value has to be all or a .csv file"
    echo ""
    exit 1
  else 
    if [ ! -f "$PATH_PROJECT/input/$dataset" ]; then
      echo "The file $dataset does not exist in the input folder"
      echo ""
      exit 1
    fi
  fi
fi

sed -i "s/^spark.data.input_file=.*/spark.data.input_file=$dataset/" "$SPARK_PROPERTIES"
sed -i "s/^spark.pipeline.active_exploratory_analysis=.*/spark.pipeline.active_exploratory_analysis=$exploratory_analysis/" "$SPARK_PROPERTIES"
sed -i "s/^spark.pipeline.active_preprocess_job=.*/spark.pipeline.active_preprocess_job=$preprocess/" "$SPARK_PROPERTIES"
sed -i "s/^spark.pipeline.active_fpr_fss=.*/spark.pipeline.active_fpr_fss=$fpr_fss/" "$SPARK_PROPERTIES"
sed -i "s/^spark.pipeline.active_fdr_fss=.*/spark.pipeline.active_fdr_fss=$fdr_fss/" "$SPARK_PROPERTIES"
sed -i "s/^spark.pipeline.active_fwe_fss=.*/spark.pipeline.active_fwe_fss=$fwe_fss/" "$SPARK_PROPERTIES"
sed -i "s/^spark.pipeline.active_train_job=.*/spark.pipeline.active_train_job=$train/" "$SPARK_PROPERTIES"

echo "The execution will be done with the following options."
echo "=================================================================="
echo "Dataset(s).......................: $dataset"
echo "Exploratory analysis.............: $exploratory_analysis"
echo "Train preprocess pipeline........: $preprocess"
echo "Train Univariate fss (fpr).......: $fpr_fss"
echo "Train Univariate fss (fdr).......: $fdr_fss"
echo "Train Univariate fss (fwe).......: $fwe_fss"
echo "Train models fss.................: $train"
echo "=================================================================="
echo ""

#spark-submit \
#    --properties-file $SPARK_PROPERTIES \
#    --master local[*] \
#    --py-files $APP_NAME-python.zip \
#    $PATH_PROJECT/src/main/python/core/main.py