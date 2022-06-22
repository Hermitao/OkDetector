# Linha que irá ser adicionada, que nós iremos pegar as coordenadas da mão
novaLinha = ['Exemplo, de, linha, nova']

#----------------------------
# Aqui ele abri um arquivo txt, criado previamente, e adiciona a linha nova
with open('C:/Users/joaop/OneDrive/Documents/garotoDePrograma/Python/py/OkDetector-main/TXTDados/base.txt', 'a') as file:
    file.write('\n')
    for x in novaLinha:
        file.write(x)
#----------------------------

#----------------------------
# Aqui ele mostra toda a base de dados
with open('C:/Users/joaop/OneDrive/Documents/garotoDePrograma/Python/py/OkDetector-main/TXTDados/base.txt', 'r') as file:
    for x in file:
        print(x, end='')
#----------------------------

# Eu só consegui fazer com caminho absoluto, quem conseguir arrumar e deixar com relativo ganha uma mamadinha. Sério, esse negócio tá me deixando maluco...