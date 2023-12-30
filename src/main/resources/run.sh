#!/bin/bash

. ./run-env.sh

dataset=$(grep "^spark\.data\.default_input_file=" "$SPARK_PROPERTIES" | awk -F '=' '{print $2}')
exploratory_analysis=$(grep "^spark\.pipeline\.default_active_exploratory_analysis=" "$SPARK_PROPERTIES" | awk -F '=' '{print $2}')
train=$(grep "^spark\.pipeline\.default_active_train_job=" "$SPARK_PROPERTIES" | awk -F '=' '{print $2}')


echo "                                  _             ____  _  ___      "
echo " _ __ _    _ ___ _ __   __ _ _ __| | __        / ___|| ||__ \     "
echo "| '_ \ \  / / __| '_ \ / _\` | '__| |/ /  ___  | |  _|| |  ) |   "
echo "| |_) \ \/ /\__ \ |_) | (_| | |  |   <  |___| | |_| || | / /    "
echo "| .__/ \  / |___/ .__/ \__,_|_|  |_|\_\        \____||_||____|  "
echo "|_|    /_/      |_|                                             "
echo "=================================================================="


while getopts ":d:e:t:" flag; do
 case $flag in
   d) dataset=${OPTARG};;
   e) exploratory_analysis=${OPTARG};;
   t) train=${OPTARG};;
   \?)
     echo "The flag introduced is not valid. Execution stopped"
     exit 1
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

if [ ! "$exploratory_analysis" = "true" ] && [ ! "$exploratory_analysis" = "false" ]; then
  echo "The input -e value has to be true or false"
  echo ""
  exit 1
fi

sed -i "s/^spark.data.input_file=.*/spark.data.input_file=$exploratory_analysis/" "$SPARK_PROPERTIES"

if [ ! "$train" = "true" ] && [ ! "$train" = "false" ]; then
  echo "The input -t value has to be true or false"
  echo ""
  exit 1
fi

sed -i "s/^spark.data.input_file=.*/spark.data.input_file=$train/" "$SPARK_PROPERTIES"

echo "The execution will be done with the following options."
echo "=================================================================="
echo "Dataset(s)..........: $dataset"
echo "Exploratory analysis: $exploratory_analysis"
echo "Train models........: $train"
echo "Otra opcion.........: Enrique putero"
echo "=================================================================="
echo ""

spark-submit \
    --properties-file $SPARK_PROPERTIES \
    --master local[*] \
    --py-files $APP_NAME-python.zip \
    $PATH_PROJECT/src/main/python/core/main.py