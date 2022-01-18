ssh node10.bdcl "sudo rm -rf /mnt/ess_storage/DN_1/storage/sber_LAMA/code/*" && \
scp -r lightautoml tests node10.bdcl:/mnt/ess_storage/DN_1/storage/sber_LAMA/code/ && \
ssh node10.bdcl "sudo chmod 777 -R /mnt/ess_storage/DN_1/storage/sber_LAMA/code"
