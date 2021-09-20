SCRIPT=run_fedavg_distributed_pytorch.sh
PY=python3


SRC=./main_fedavg.py


CLIENT_NUM=2
WORKER_NUM=2
MODEL=lr
DISTRIBUTION=hetero
ROUND=1
EPOCH=1
BATCH_SIZE=10
LR=0.03
DATASET=mnist
DATA_DIR=./../../../data/mnist
CLIENT_OPTIMIZER=sgd
CI=0

PROCESS_NUM=$$(( $(WORKER_NUM) + 1 ))

run:
	mpirun -np $(PROCESS_NUM) -hostfile ./mpi_host_file $(PY) $(SRC) \
		--model $(MODEL) \
		--dataset $(DATASET) \
		--data_dir $(DATA_DIR) \
		--partition_method $(DISTRIBUTION)  \
		--client_num_in_total $(CLIENT_NUM) \
		--client_num_per_round $(WORKER_NUM) \
		--comm_round $(ROUND) \
		--epochs $(EPOCH) \
		--client_optimizer $(CLIENT_OPTIMIZER) \
		--batch_size $(BATCH_SIZE) \
		--lr $(LR) \
		--ci $(CI) 
