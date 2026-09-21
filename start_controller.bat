SET basePath=%cd%
SET controllerName=fc-controller

mkdir data

docker kill %controllerName%
docker rm %controllerName%
docker network rm %controllerName%-internal
docker pull featurecloud.ai/controller

docker run ^
-d ^
-p 8006:8000 ^
--name %controllerName% ^
-v "/var/run/docker.sock:/var/run/docker.sock" ^
--mount "type=bind,source=%basePath%/data,target=/data" ^
featurecloud.ai/controller "--host-root=%basePath%/data" "--internal-root=/data" "--controller-name=%controllerName%"
