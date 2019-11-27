from utils import measure_rmse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def OLS_plot_res(model, results, train, test=None, _Save=False) :
    train_input, train_output = train[0], train[1]
    y_hat = model.predict(params=results.params, exog=train_input)
    rmse = measure_rmse(y_hat, train_output)
    print('Root Mean Squared Error on Train : {}'.format(rmse))

    if test :
        test_input, test_output = test[0], test[1]
        yhat_test = model.predict(params=results.params, exog=test_input)
        rmse_test = measure_rmse(yhat_test, test_output)
        print('Root Mean Squared Error on test : {}'.format(rmse_test))

    return 0

def NN_plot_res(model, history, train, test=None, _Save=False):
    gs = gridspec.GridSpec(4, 2)
    ax1 = plt.subplot(gs[:,0])
    ax2 = plt.subplot(gs[0,1])
    ax3 = plt.subplot(gs[1,1])
    ax4 = plt.subplot(gs[2,1])
    ax5 = plt.subplot(gs[3,1])


    # Residualual plot


    print(history.history.keys())

    ax4.plot(history.history['mean_absolute_error'])
    ax4.plot(history.history['val_mean_absolute_error'])
    ax4.set_title('model accuracy')
    ax4.set_ylabel('accuracy')
    ax4.set_xlabel('epoch')
    ax4.legend(['train', 'test'])

    ax5.plot(history.history['loss'])
    ax5.plot(history.history['val_loss'])
    ax5.set_title('loss accuracy')
    ax5.set_ylabel('loss')
    ax5.set_xlabel('epoch')
    ax5.legend(['train', 'test'])

    plt.show()

    return 0
