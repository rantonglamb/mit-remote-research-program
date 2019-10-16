# MNIST DOCKER CASSANDARA
please try to run this with UNIX or LINUX,Windows is not recommended. I myself had tried to work on windows7, I have experienced so many 
unexpected bugs,yet nothing went wrong when I tried to run it on a computer with ubantu system in my former university .And the internet 
in the mainland of China could also cause problems while proceeding to the second step when you want to build image for the app.I recommend
you either try to run it in a formal computer lab in an university or set the downloading route to the image net routes,which I mentioned in the 
requirement. 
##description
Since the mnist technique is already well developed ,I just saved the model as a ckpt file.Then, via Flask, a well-known light
weight web application framework,we could generate a web on which we could upload a picture of a single digit,(pay attention to the size of
the pictures ,which should be pre-prepared if it is not 28*28 or it will cause problems),the recognization result of the picture will then 
be returned.Need to mention that the user could upload the picture by two means:uploading it from  a browser,or upload it to a route that 
we assign in the app.

In the following step,we are going to construct two conatainers,one is to contain the app that we have computed,and the other is the one which 
could contain the Cassandra database.These two containers are connected by a docker network bridge.

Here are the codes which will be needed to achieve that:
1 construct a docker network in order to connect the containers

docker network create [name]


2 build the image for the app with the Dockerfile:
docker build -t [name]:latest . (do not forget about the dot)


3 Pull the image of cassandra
docker pull cassandra 

3 build a container for cassandra
docker run --name cassandra --net=[name] -p 9042:9042 -d cassandra:latest 

4 build the container for the app image
docker run --name [app image] --net=[name] -d -p 8000:5000 :latest

5use curl to upload and check the result
curl -X POST -F image=@[path] '[url]'

6 check the information you have stored in the cassandra data base 

6docker exec -it cassandra cqlsh


##problems you may encounter
as I have mentioned before,you are required to build the image for the app with your dockerfile,the fact is I ran the same 
code suuccessfully in a lab in a univerisity,but it never worked out when I tried it at home.That may be caused by the internet.(each
time I ran,it was shown time out)

