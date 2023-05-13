from training import main

if __name__ == '__main__':

    models = ['FNDE_2', 'FNDE_2_mod', 'FNO', 'NODE']
    for model in models:
        for i in range(0,5):
            main(model_name = model, exp="000", repeat=f'{i}___')
    