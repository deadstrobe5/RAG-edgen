.PHONY: all check_qdrant start_qdrant check_unstructured start_unstructured start_rag

all: start_qdrant start_unstructured start_rag

check_qdrant:
	@echo "Checking if Qdrant is up..."
	@counter=0; \
	while ! nc -z localhost 6333; do \
		sleep 1; \
		counter=$$((counter+1)); \
		if [ $$counter -eq 10 ]; then \
			echo "Still waiting for Qdrant to start..."; \
			counter=0; \
		fi; \
	done
	@echo "Qdrant is up and running."

start_qdrant: check_qdrant
	@echo "Starting Qdrant (port 6333) ..."
	@docker run -p 6333:6333 qdrant/qdrant > /dev/null 2>&1 &

check_unstructured:
	@echo "Checking if Unstructured-API is up..."
	@if [ ! -d "unstructured-api" ]; then \
		echo "Installing Unstructured-API..."; \
		git clone https://github.com/Unstructured-IO/unstructured-api > /dev/null 2>&1; \
		cd unstructured-api && git pull > /dev/null 2>&1; \
		make install > /dev/null 2>&1; \
	fi

start_unstructured: check_unstructured
	@echo "Starting Unstructured-API (port 8000) ..."
	@cd unstructured-api && make run-web-app > /dev/null 2>&1 &
	@echo "Waiting for Unstructured-API to be up..."
	@counter=0; \
	while ! nc -z localhost 8000; do \
		sleep 1; \
		counter=$$((counter+1)); \
		if [ $$counter -eq 10 ]; then \
			echo "Still waiting for Unstructured-API to start..."; \
			counter=0; \
		fi; \
	done
	@echo "Unstructured-API is up and running."

start_rag:
	@echo "Starting RAG server..."
	@python3 rag-server.py
