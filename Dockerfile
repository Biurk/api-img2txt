#FROM continuumio/miniconda3
#WORKDIR /usr/src/app
#COPY environment.yml .
#RUN apt-get update && apt-get -y install nodejs && apt-get -y install npm && \
#conda update -n base -c defaults conda && conda env create -f environment.yml && npm i npm@6.4.1
#COPY . .
#RUN npm i
#ENTRYPOINT ["npm", "run", "start"]

FROM node
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y python3-pip && pip3 install numpy
COPY package*.json ./
RUN npm install
COPY . .
CMD ["npm","start"]
