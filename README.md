# Instruções de execução
1. Na pasta raiz do projeto, crie um ambiente de desenvolvimento python com python -m venv env. Caso ele peça para baixar, utilize o comando `pip install --user virtualenv`
2. Quando finalizado, inicie o ambiente de desenvolvimento com `source env/bin/activate`, caso esteja no Linux ou `source env/Scripts/activate`, caso esteja no Windows.
3. Instale as dependências do projeto com `pip install -r requirements.txt`
4. Entre na pasta PAI_TRAB01
5. Execute o comando `./scripts/migrate_and_run.sh`, em um ambiente linux. Caso não esteja em um ambiente linux, execute o comando `python3 manage.py runserver`.
6. Acesse o site em http://localhost:8000

# Observações
* Na tela de comparar, cada vez que for executar a função de comparar, reinsira as duas imagens