# DEEV: Dispositivos, Eu Escolho Vocês: Seleção de Clientes Adaptativa para Comunicação Eficiente em Aprendizado Federado

**Resumo**: *O aprendizado federado (Federated Learning -- FL) é uma abordagem distribuída para o treinamento colaborativo de modelos de aprendizado de máquina. O FL requer um alto nível de comunicação entre os dispositivos e um servidor central, assim gerando diversos desafios, incluindo gargalos de comunicação e escalabilidade na rede. Neste trabalho, introduzimos DEEV, uma solução para diminuir os custos gerais de comunicação e computação para treinar um modelo no ambiente FL. DEEV emprega uma estratégia de seleção de clientes que adapta dinamicamente o número de dispositivos que treinam o modelo e o número de rodadas necessárias para atingir a convergência. Um caso de uso no conjunto de dados de reconhecimento de atividades humanas é realizado para avaliar DEEV e compará-lo com outras abordagens do estado da arte. Avaliações experimentais mostram que DEEV reduz eficientemente a sobrecarga geral de comunicação e computação para treinar um modelo e promover sua convergência. Em particular, o DEEV reduz em até 60% a comunicação e em até 90% a sobrecarga de computação em comparação com as abordagens da literatura, ao mesmo tempo em que fornece boa convergência mesmo em cenários em que os dados são distribuídos de forma não independente e idêntica entre os dispositivos clientes.*

## Datasets disponíveis

Os seguintes datasets estão disponíveis para a avaliação:
- CIFAR-10
- MNIST
- Motion sense
- UCI-HAR

## Parâmetros para gerar docker-compose-file:
- `--clients` `-c`: Quantidade total de clientes
- `--model` `-m`: Modelo de ML/DL para ser utilizado no treinamento (e.g., DNN, CNN, or Logistic Regression)
- `--client-selection` `-`: Método para seleção de clientes (e.g., POC, DEEV)
- `--dataset` `-d`: Dataset para ser utilizado no treinamento (e.g., MNIST, CIFAR10)
- `--local-epochs` `-e`:  Quantidade de épocas locais de treinamento
- `--rounds` `-r`: Número de rodadas de comunicação para o treinamento
- `--poc` `-`: Porcentagem de clientes para ser selecionados no Power-of-Choice
- `--decay` `-`: Parâmetros para decaimento no DEEV

## Criando arquivo de configuração:
```python
python create_dockercompose.py --client-selection='DEEV' --dataset='MNIST' 
--model='DNN' --epochs=1 --round=10 --clients=50 
```

## Como executar
```shell
docker compose -f <compose-file.yaml> --compatibility up 
```
## Como citar
@inproceedings{deev,
 author = {Allan M. de Souza and Luiz Fernando Bittencourt and Eduardo Cerqueira and Antonio Loureiro and Leandro A. Villas},
 title = {Dispositivos, Eu Escolho Vocês: Seleção de Clientes Adaptativa para Comunicação Eficiente em Aprendizado Federado},
 booktitle = {Anais do XLI Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos},
 location = {Brasilia},
 year = {2023},
 keywords = {},
 issn = {},
 pages = {},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {},
 url = {}
}

