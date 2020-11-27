# Summarisation API

A docker container for running the summarisation API 

## GPU

### To install
```
nvidia-docker build -t summarisation_api .
nvidia-docker run -p 5000:5000 --name summarisation_api -i -t summarisation_api
exit()
```

### To restart
```
nvidia-docker stop summarisation_api
nvidia-docker rm summarisation_api
nvidia-docker build -t summarisation_api .
nvidia-docker run -p 5000:5000 --name summarisation_api -i -t summarisation_api
```

### To run
```
nvidia-docker start summarisation_api
nvidia-docker exec -it summarisation_api /bin/bash
```

## CPU 


### To install
```
docker build -t summarisation_api .
docker run -p 5000:5000 --name summarisation_api -i -t summarisation_api
exit()
```

### To restart
```
docker stop summarisation_api
docker rm summarisation_api
docker build -t summarisation_api .
docker run -p 5000:5000 --name summarisation_api -i -t summarisation_api
```

### To run
```
docker start summarisation_api
docker exec -it summarisation_api /bin/bash
```