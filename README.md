# neuralNetworkOnKotlin  
Консольная реализация.  
Простейшая нейронная сеть(3 входных нейрона, 2 внутренних нейрона, 1 выходной нейрон), функция активации - сигмоид.  
На вход передаётся набор данных(trainingData), а так же ожидаемое предсказание(expectedValue), для этого конкретного набора данных.  
  
Функция main() ->  
Сначала выводится состояние сети до обучения, а так же предсказания на основе этого состояния, после происходит обучение...  
Затем еще раз выводится состояние сети, только уже после обучения и соответсвенно предсказания на основе обученной сети.  
  
По умолчанию я закоментировал всю детализацию процесса обучения и вывод "mean square error" после каждой эпохи,  
для просмотра достаточно раскоментировать блоки кода помеченный как "DETAILS".  
При это рекомендую закоментировать вывод состояния сети и результаты предсказания - блоки кода помеченные как "OUTPUT ON CONSOLE".
