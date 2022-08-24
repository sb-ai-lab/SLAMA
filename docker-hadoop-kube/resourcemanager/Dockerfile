ARG IMAGE_TAG=slama-yarn
FROM node2.bdcl:5000/yarn-base:${IMAGE_TAG}

MAINTAINER Ivan Ermilov <ivan.s.ermilov@gmail.com>

HEALTHCHECK CMD curl -f http://localhost:8088/ || exit 1

ADD run.sh /run.sh
RUN chmod a+x /run.sh

EXPOSE 8088

CMD ["/run.sh"]
