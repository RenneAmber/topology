#!/bin/bash
#!/home/chip/anaconda3/envs/py37/bin/python
#./main.sh -n lenet -d mnist -t 20 -e 50 -g '1 2 3 4 5 7 10 15 20 30 40 50' -p 0 -s 1 -u 1 -l 0.0001 &

function usage {
    echo "usage: $main [-net network_name] [-dt dataset_name] [-trl trial] [-netr n_epochs_train] [-ete epochs_test] [-plb permute_labels] [-dtsub datasubset]"
    echo "  -n  Specify deep network architecture (e.g. lenet, alexnet, resnet, inception, vgg, etc)"
    echo "  -d   Specify dataset (e.g. mnist, cifar10, imagenet)"
    echo "  -t    Specify trial number. Used to differentiate btw multiple trainings of same setup."
    echo "  -e   Specify number of training epochs. "
    echo "  -g    Specify list of epochs for which graph building is going to be performed. Sequence of positive integers delimited by blank space."
    echo "  -p    Specify if labels are going to be permuted. Float between 0 and 1. If 0, no permutation. If 1 all labels are permuted. Otherwise proportion of labels."
    echo "  -s  Specify if subset of data should be loaded. Float between 0 and 1. If 0, all data, else proportio of data randomly sampled. "
    echo "  -u  Specify the function type of generating function. In {0,1}. 0 -- get function, 1 -- get structure"
    exit 1
}


while getopts n:d:t:e:g:p:s:u:l: option
do
    case "${option}"
    in
	n) NET=${OPTARG};;
	d) DATASET=${OPTARG};;
	t) TRIAL=${OPTARG};;
	e) N_EPOCHS_TRAIN=$OPTARG;;
	g) EPOCHS_TEST=$OPTARG;;
	p) PERM_LABELS=$OPTARG;;
	s) DATA_SUBSET=$OPTARG;;
    u) FUNCTION_TYPE=$OPTARG;;
    l) LEARNING_RATE=$OPTARG;;
    esac
done

#THRESHOLDS="0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50"
THRESHOLDS=()
for i in {0..7} 
do
    THRESHOLDS="$THRESHOLDS 0`echo "scale=3;0.200+0.100*$i"|bc -l`"
done

for pp in {0..4}
do
CURR_TRIAL=$((TRIAL+pp))
echo $CURR_TRIAL
echo ""
echo "----------------------------------------"
echo "Training network"
echo "----------------------------------------"
echo ""
python ../train.py --net $NET --dataset $DATASET --trial $CURR_TRIAL --epochs $N_EPOCHS_TRAIN --permute_labels $PERM_LABELS --subset $DATA_SUBSET --lr $LEARNING_RATE

echo ""
echo "----------------------------------------"
echo "Building graph"
echo "----------------------------------------"
echo ""
if [$FUNCTION_TYPE -eq '0']; then
    python ../build_graph_functional.py --net $NET --dataset $DATASET --trial $CURR_TRIAL --epochs $EPOCHS_TEST --thresholds $THRESHOLDS
else
    python ../build_graph_structural.py --net $NET --dataset $DATASET --trial $CURR_TRIAL --epochs $EPOCHS_TEST --thresholds $THRESHOLDS --function_type $FUNCTION_TYPE
fi
echo ""
echo "----------------------------------------"
echo "Computing topology"
echo "----------------------------------------"
echo ""

PATH_NAME="../../data/adjacency/"
for e in $EPOCHS_TEST
do
    for t in $THRESHOLDS
    do
        eval "./cpp/symmetric "$PATH_NAME$NET"_"$DATASET"/badj_epc"$e"_t"$t"_trl"$CURR_TRIAL".csv 1 0"	
    done
done

echo ""
echo "----------------------------------------"
echo "Prepare topology results"
echo "----------------------------------------"
echo ""

python prepare_results.py --net $NET --dataset $DATASET --trial $CURR_TRIAL --epochs $EPOCHS_TEST  --thresholds $THRESHOLDS --permute_labels $PERM_LABELS --subset $DATA_SUBSET

# python draw_image.py --net $NET --dataset $DATASET --trial $CURR_TRIAL
done
