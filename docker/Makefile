IMAGE_NAME=nube266/ros-detectron2
CONTAINER_NAME=ros-detectron2
COMPOSE_ARGS=CONTAINER_NAME=$(CONTAINER_NAME) \
			 IMAGE_NAME=$(IMAGE_NAME)
COMPOSE_FILE=docker-compose.yml

init: build up ## docker-compose build && docker-compose up -d

setid: ## set container id
	$(eval CONTAINER := $(shell docker ps -aqf 'name=$(CONTAINER_NAME)' ))

build: ## docker-compose build
	@$(COMPOSE_ARGS) docker-compose -f $(COMPOSE_FILE) build

up: setid ## docker-compose up -d
	@$(COMPOSE_ARGS) docker-compose -f $(COMPOSE_FILE) up -d

down: setid
	@$(COMPOSE_ARGS) docker-compose -f $(COMPOSE_FILE) down

bash: setid ## docker exec -it CONTAINER /bin/bash
	# @xhost +local:user
	@docker exec -it $(CONTAINER_NAME) /bin/bash

help: ## show this help
	@echo "make subcommand list:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

-h: ## show this help
	@echo "make subcommand list:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

-help: ## show this help
	@echo "make subcommand list:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

--help: ## show this help
	@echo "make subcommand list:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'