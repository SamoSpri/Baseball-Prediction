#3차원 가공 모델 돌리고 결과뽑는거
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pandas as pd
import timeit

TRAIN_URL = 'transformed_lstm_train_data.csv'
TEST_URL = 'transformed_lstm_test_data.csv'


# tf.layers.dense 를 이해하기 위한 함수 정의
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        # X 는 (한 배치의 데이터 수, 입력 특성의 수)
        n_inputs = int(X.get_shape()[1])
        # 표준편차가 다음과 같은 정규분포(절단 정규분포)를 사용하여, 가중치 행렬(커널) W 를 무작위 초기화
        # 절단 정규분포를 사용하면 큰 가중치가 생기지 않는다
        # 가중치가 크면 계산된 출력값이 커져, 로지스틱 함수의 양극단에 가까워진다
        # 기울기가 0에 가까워짐에 따라 오차 그래디언트는 희미해지고, 가중치 업데이트가 느려진다
        # truncated_normal 이라는 함수는 2 표준편차 이상의 값을 제외한 정규분포 난수를 발생시키는 함수
        stddev = 2 / np.sqrt(n_inputs + n_neurons)

        # W 는 (입력 특성의 수, 노드의 수) 모양을 갖는다
        # init : 정의된 정규분포에 따라 가중치 행렬 초기화
        init = tf.random.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        # constant_weight : 고정된 동일한 값으로 가중치 행렬 초기화
        constant_weight = tf.constant(0.4, shape=[n_inputs, n_neurons])

        W = tf.Variable(init, name='kernel')
        b = tf.Variable(tf.zeros([n_neurons]), name='bias')
        # Z 는 (한 배치의 데이터 수, 노드의 수) 모양을 갖는다
        Z = tf.matmul(X, W) + b

        if activation is not None:
            return activation(Z)
        else:
            return Z


# 배치를 하나씩, 다음 호출시에는 다음 배치를 반환
# shape : (전체 게임 수, 특징 수) -> (batch_size * date_num(time_step), 특징 수)
# batch_size 는 하루에 일어나는 경기 수 만큼
def make_one_next_batch(X, y, batch_size, time_step):
    index = 0
    batch_num = int((X.shape[0] - time_step * batch_size) / batch_size) + 1
    # batch_num = int((X.shape[0] - time_step * batch_size) / batch_size / 2) + 1
    for _ in range(batch_num):
        X_batch, y_batch = X[index: index + batch_size * time_step], y[index: index + batch_size * time_step]
        index += batch_size
        # index += batch_size * 2
        if X_batch.shape[0] == batch_size * time_step:
            yield X_batch, y_batch
        else:
            print('batch shape 0 :', X_batch.shape[0])
            print('batch size * time step :', batch_size * time_step)
            print('shape 불일치')
            break


# 하루에 일어난 경기 수 범위내에서 배치 순서 무작위 변환
# X, y 의 shape 은 (batch_size, data_num(time_step), 특징 수)
def shuffle_game_order(X, y, batch_size):
    shuffled_batch_index = np.random.permutation(batch_size)
    return X[shuffled_batch_index], y[shuffled_batch_index]


# input 과 target 을 batch 크기만큼 잘라, batch_step_term 만큼 건너뛰며 배치 생성
def make_batch_set(X, y, batch_size, batch_step_term):
    X_batch = X[0: batch_size]
    y_batch = y[0: batch_size]
    # axis 할당 안하면 데이터가 1자로 펴짐
    for batch_idx in range(batch_step_term, X.shape[0] - batch_size + 1, batch_step_term):
        X_batch = np.append(X_batch, X[batch_idx: batch_idx + batch_size], axis=0)
        y_batch = np.append(y_batch, y[batch_idx: batch_idx + batch_size], axis=0)
    return X_batch, y_batch


def load_data():
    train_path, test_path = TRAIN_URL, TEST_URL

    targets = []
    for index in range(10):
        targets.append('win_' + str(index))
    train = pd.read_csv(train_path)

    train_y = train.pop(targets[0])
    for target in targets[1:]:
        train_y = pd.concat((train_y, train.pop(target)), axis=1)
    train_x = train

    test = pd.read_csv(test_path)
    test_y = test.pop(targets[0])
    for target in targets[1:]:
        test_y = pd.concat((test_y, test.pop(target)), axis=1)
    test_x = test

    # 경기 정보를 한글로 갖는 데이터
    k_test_df = pd.read_csv('transformed_lstm_test_data_k.csv')

    # 하루에 진행된 경기가 5 건이 아닌 경우 경우,
    # 진행되지 않은 경기에 대해 공백으로 처리하여 모양을 맞추거나
    # 두 번 이상 진행된 경기 정보 제거

    train_date_list = train_x['b0'].unique()
    test_date_list = test_x['b0'].unique()
    for date in train_date_list:
        date_game_num = int(len(train_x[train_x['b0'] == date]) / 2)
        if date_game_num < 5:
            # 부족한 경기 수 만큼 진행되지 않은 경기로 정보를 채움
            # 정보가 채워질 index 위치(값) list
            index = train_x[train_x['b0'] == date].index.values[0] + date_game_num
            insert_loc_index_list = []
            # 선발 선수만의 경기 1번, 전체 선수 경기 1번 총 두 번의 경기 정보 추가 필요
            for _ in range(2):
                for i in range(date_game_num, 5):
                    insert_loc_index_list.append(index)
                    index += 1
                index += date_game_num

            # 채워질 정보 생성
            to_insert_row_X = pd.DataFrame(columns=train_x.columns, data=0, index=[0])
            to_insert_row_y = pd.DataFrame(columns=train_y.columns, data=-1, index=[0])

            # 채울 index 만큼 반복하며 정보를 채움
            for insert_loc_index in insert_loc_index_list:
                train_x = pd.concat([train_x.iloc[:insert_loc_index], to_insert_row_X, train_x.iloc[insert_loc_index:]])
                train_x.reset_index(drop=True, inplace=True)
                train_y = pd.concat([train_y.iloc[:insert_loc_index], to_insert_row_y, train_y.iloc[insert_loc_index:]])
                train_y.reset_index(drop=True, inplace=True)

            # print(train_x.loc[train_x[train_x['b0'] == date][-1:].index.values[0] - 9: train_x[train_x['b0'] == date][-1:].index.values[0] + 9, :])

    for date in test_date_list:
        date_game_num = int(len(test_x[test_x['b0'] == date]) / 2)
        if date_game_num < 5:
            # 부족한 경기 수 만큼 진행되지 않은 경기로 정보를 채움
            # 정보가 채워질 index 위치(값) list
            index = test_x[test_x['b0'] == date].index.values[0] + date_game_num
            insert_loc_index_list = []
            # 선발 선수만의 경기 1번, 전체 선수 경기 1번 총 두번의 경기 정보 추가 필요
            for _ in range(2):
                for i in range(date_game_num, 5):
                    insert_loc_index_list.append(index)
                    index += 1
                index += date_game_num

            # 채워질 정보 생성
            to_insert_row_X = pd.DataFrame(columns=test_x.columns, data=0, index=[0])
            to_insert_row_y = pd.DataFrame(columns=test_y.columns, data=-1, index=[0])
            to_insert_row_k = pd.DataFrame(columns=k_test_df.columns, data=0, index=[0])

            # 부족한 경기 수 만큼 반복하며 정보를 채움
            for insert_loc_index in insert_loc_index_list:
                test_x = pd.concat([test_x.iloc[:insert_loc_index], to_insert_row_X, test_x.iloc[insert_loc_index:]])
                test_x.reset_index(drop=True, inplace=True)
                test_y = pd.concat([test_y.iloc[:insert_loc_index], to_insert_row_y, test_y.iloc[insert_loc_index:]])
                test_y.reset_index(drop=True, inplace=True)
                k_test_df = pd.concat(
                    [k_test_df.iloc[:insert_loc_index], to_insert_row_k, k_test_df.iloc[insert_loc_index:]])
                k_test_df.reset_index(drop=True, inplace=True)

            # print(test_x.loc[test_x[test_x['b0'] == date][-1:].index.values[0] - 9 : test_x[test_x['b0'] == date][-1:].index.values[0] + 9, :])

    # 날짜 제거
    train_x.drop('b0', axis=1, inplace=True)
    test_x.drop('b0', axis=1, inplace=True)

    # values 를 할당하여 dataframe 에서 ndarray 로 변경
    train_x, train_y = train_x.values, train_y.values.astype(int)
    test_x, test_y = test_x.values, test_y.values.astype(int)

    return (train_x, train_y), (test_x, test_y, k_test_df)


if __name__ == '__main__':
    # 데이터 로드
    # 로드된 데이터 shape 은 (전체 게임 수, 특징 수), (전체 게임 수, output 수)
    (X_train, y_train), (X_test, y_test, k_test_df) = load_data()

    # 학습할 일자 수 * 2
    max_date_num = 60 * 2

    if max_date_num < X_train.shape[0]:
        date_num = max_date_num
    else:
        date_num = X_train.shape[0] // 5

    # 특징(선수) 수
    feature_num = X_train.shape[1]

    hidden_units = [100, 50]

    n_outputs = 10

    # input X 의 shape 은 (batch size, 학습 경기 수(time_step), 특징(선수) 수)
    X = tf.compat.v1.placeholder(tf.float32, [None, date_num, feature_num], name='X')
    y = tf.compat.v1.placeholder(tf.float32, [None, date_num, n_outputs])
    keep_prob = tf.compat.v1.placeholder_with_default(1.0, shape=())

    lstm_layers = []
    for units in hidden_units:
        layer = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=units, activation=tf.nn.relu)#, use_peepholes=True)
        # 각 층에 drop out 적용
        drop_layer = tf.compat.v1.nn.rnn_cell.DropoutWrapper(layer, input_keep_prob=keep_prob)

        lstm_layers.append(drop_layer)

    # dynamic_rnn 은 입력 X의 값 중, 두 번째 차원의 값(step 수) 만큼 while loop을 돌며 행렬곱 연산 생성
    # rnn_outputs 의 shape 은 (batch 크기, date_num, hidden units)
    multi_layer_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_layers)
    rnn_outputs, states = tf.compat.v1.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    # 생성된 output을 배열로 쌓은 후 완전연결층을 통과시켜 마지막 차원(뉴런 수 만큼 나온 output)을 줄임
    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, hidden_units[-1]])
    stacked_outputs = tf.compat.v1.layers.dense(stacked_rnn_outputs, n_outputs, activation=tf.nn.tanh)

    outputs = tf.reshape(stacked_outputs, [-1, date_num, n_outputs], name='outputs')

    loss = tf.reduce_mean(tf.square(outputs - y))

    # optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=10)
    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    train_step = 1000
    train_keep_prob = 0.9

    pred_index = 0
    result_index = 0
    batch_size = 5
    result = []

    total_pred_count = 0
    total_correct_pred_count = 0
    total_canceled_game_count = 0

    # 경기 결과(output) 한글표현 열
    column_list = list(k_test_df.columns.values)
    win_info_list = []
    for col in column_list:
        if 'win' in col:
            win_info_list.append(col)

    # 예측을 요약할 데이터 프레임
    result_df = pd.DataFrame(
        columns=['날짜', 'home team', 'away team', 'home team 승리 확률', 'away team 승리 확률', '예측 승리 팀', '실제 승리 팀', '적중 여부'])

    # 예측의 통계를 담을 데이터프레임. 전체 예측 시도 수와 각 예측율의 등장 횟수 집계
    stoc_result = pd.DataFrame(columns=['예측 시도 일 수'], index=[0], data=0)

    result_df['날짜'] = k_test_df['날짜']

    # 예측을 시도할 경기 수 만큼 학습&예측 반복
    while pred_index < y_test.shape[0]:
        # 학습
        with tf.compat.v1.Session() as sess:
            init.run()

            train_start_time = timeit.default_timer()
            for step in range(train_step):
                mse_sum = 0
                step_start_time = timeit.default_timer()
                # 학습 데이터 shape 변환. (전체 게임 수, 특징 수) -> (배치 수 * date_num(time_step), 특징 수)
                # make_one_next_batch 를 통해 하루에 일어나는 경기 수(5 경기) 만큼 한 배치 생성 & 반환
                for X_batch, y_batch in make_one_next_batch(X_train, y_train, batch_size, date_num):
                    # shape 재 변환. (배치 수 * date_num(time_step), 특징 수) -> (date_num(time_step), 배치 수 ,특징 수)
                    # shape 변환 순서는 1. 1자로 펴기, 2. 특징 수로 나누기, 3. 배치 수로 나누기, 4. date_num 으로 나누어 떨어지는지 확인
                    # 처음부터 배치 수, date_num 순서로 reshape 하면 한 배치의 한 스텝에 여러 날짜가 할당됨
                    X_batch = X_batch.reshape(date_num, -1, feature_num)
                    y_batch = y_batch.reshape(date_num, -1, n_outputs)

                    # 배치 수와 date_num 의 순서를 바꿔줌
                    # 두 번의 절차를 거쳐, 한 배치의 한 스텝에는 하루에 일어난 경기 5팀에 대한 정보가 모임
                    # 최종 shape 은 (배치 수, date_num(time_step), 특징 수)
                    # time_step 은 선발 선수 경기 -> 전체 선수 경기 -> 다음날 선발 선수 경기 -> 다음날 전체 선수 경기
                    # 배치 첫 번째 차원은 하루에 일어난 경기 수 + 취소된 경기 수
                    X_batch = np.transpose(X_batch, axes=[1, 0, 2])
                    y_batch = np.transpose(y_batch, axes=[1, 0, 2])

                    # 입력을 넣기 전에 배치 순서(하루에 일어난 게임 정보 순서)를 무작위 변환
                    X_batch, y_batch = shuffle_game_order(X_batch, y_batch, batch_size)

                    sess.run(train_op, feed_dict={X: X_batch, y: y_batch, keep_prob: train_keep_prob})
                    mse_sum += loss.eval(feed_dict={X: X_batch, y: y_batch})
                    batch_end_time = timeit.default_timer()

                step_end_time = timeit.default_timer()
                print('step :', step, '학습 경과 시간 : %.2f 초' % (step_end_time - train_start_time),
                      ', 현재 step 학습 시간 : %.2f 초' % (step_end_time - step_start_time))

                if step % 1 == 0:
                    # dense layer를 통과하여 나온 output 과 loss 를 평가
                    batch_num = (X_train.shape[0] - date_num * batch_size) / batch_size + 1
                    # batch_num = (X_train.shape[0] - date_num * batch_size) / batch_size / 2 + 1
                    mse = mse_sum / batch_num
                    y_pred = sess.run(outputs, feed_dict={X: X_batch, y: y_batch})
                    print('step :', step, '전체 배치 데이터 평균 제곱평균오차 :', mse)
                    correct_pred_count = 0
                    canceled_game_count = 0
                    for i in range(0, batch_size * 2):
                        for j in range(batch_size):
                            if i == 0:
                                y_result = list(y_batch[j, -(i + 1):][0])
                                # 취소된 경기일 경우 (경기 결과가 모든 팀이 참여하지 않은 것으로 기록된 경우)
                                if all([result == -1 for result in y_result]):
                                    canceled_game_count += 1
                                    continue
                                y_pred_result = y_pred[j, -(i + 1):][0]

                            else:
                                y_result = list(y_batch[j, -(i + 1):-i][0])
                                if all([result == -1 for result in y_result]):
                                    canceled_game_count += 1
                                    continue
                                y_pred_result = y_pred[j, -(i + 1):-i][0]

                            # 1 이라는 값을 갖는 곳의 index 반환
                            win_index = y_result.index(1)
                            # 0 이라는 값을 갖는 곳의 index 반환
                            lose_index = y_result.index(0)

                            # 경기에 참여한 팀의 승리에 가까운 정도를 계산
                            winning_w = (y_pred_result[win_index] + 1) / 2
                            winning_l = (y_pred_result[lose_index] + 1) / 2

                            if winning_w + winning_l != 0:
                                winning_p_w = winning_w / (winning_w + winning_l) * 100
                                winning_p_l = winning_l / (winning_w + winning_l) * 100
                            else:
                                winning_p_w = 0
                                winning_p_l = 0

                            # print('%d 일자 - %d 경기 :' % (i, j + 1), y_pred_result)
                            # print('%d 일자 - %d 경기 :' % (i, j + 1), y_result)
                            # print('승리 팀 예측 / 결과 : %d / %d'%(y_pred_result[win_index], y_result[win_index]))
                            # print('패배 팀 예측 / 결과 : %d / %d' % (y_pred_result[lose_index], y_result[lose_index]))

                            if winning_p_l < winning_p_w:
                                correct_pred_count += 1

                    print('최근 5 스텝 적중 경기 수 : %d / %d' % (correct_pred_count, batch_size * 5 * 2 - canceled_game_count))

                if mse <= 0.2 ** 2:
                    print('mse 0.2^2 이하')
                    if (batch_size * 10 - canceled_game_count) <= correct_pred_count:
                        print('최근 5 스텝 적중률 100% 이상')
                        print('5일 간 진행된 경기 수 :', batch_size * 10 - canceled_game_count, ', 취소된 경기 수 :', canceled_game_count)
                        break

            # end session train
            train_end_time = timeit.default_timer()
            print('전체 학습 경과 시간 : %.2f 초' % (train_end_time - train_start_time))
            saver.save(sess, './lstm_test')

        # 예측
        with tf.compat.v1.Session() as sess:
            saver.restore(sess, './lstm_test')

            # 예측할 데이터
            X_test_targets = X_test[pred_index: pred_index + batch_size]
            y_test_targets = y_test[pred_index: pred_index + batch_size]
            # X_test_targets = X_test[pred_index: pred_index + batch_size * 2]
            # y_test_targets = y_test[pred_index: pred_index + batch_size * 2]

            # 다음 학습에 추가할 데이터
            X_add_to_train = X_test[pred_index: pred_index + batch_size * 2]
            y_add_to_train = y_test[pred_index: pred_index + batch_size * 2]

            # step 수를 맞추기 위해 기존 데이터와 연결
            X_test_to_pred = np.concatenate([X_train, X_test_targets], axis=0)
            y_test_to_pred = np.concatenate([y_train, y_test_targets], axis=0)

            # 예측할 경기 수 만큼의 데이터만 남기고 나머지는 버림
            X_test_to_pred = X_test_to_pred[-(date_num * batch_size):]
            y_test_to_pred = y_test_to_pred[-(date_num * batch_size):]

            X_test_to_pred = X_test_to_pred.reshape(date_num, -1, feature_num)
            y_test_to_pred = y_test_to_pred.reshape(date_num, -1, n_outputs)
            X_test_to_pred = np.transpose(X_test_to_pred, axes=[1, 0, 2])
            y_test_to_pred = np.transpose(y_test_to_pred, axes=[1, 0, 2])

            # y_pred 의 shape은 (예측을 시도하는 게임의 수, date_num, n_output)
            y_pred = sess.run(outputs, feed_dict={X: X_test_to_pred})

        # end session prediction

        result.append(y_pred[:, -1, :])
        # result.append(y_pred[:, -2, :])

        correct_pred_count = 0
        canceled_game_count = 0
        # 반복문을 통해 하루 당 5 경기 씩의 결과를 출력
        for i in range(batch_size):
            total_pred_count += 1
            # 취소된 경기가 아닐 경우
            if not all([result == -1 for result in y_test_to_pred[i][-1]]):
                # if not all([result == -1 for result in y_test_to_pred[i][-2]]):

                y_test_to_pred_list = list(y_test_to_pred[i][-1])
                # y_test_to_pred_list = list(y_test_to_pred[i][-2])
                win_index = y_test_to_pred_list.index(1)
                lose_index = y_test_to_pred_list.index(0)

                # 경기에 참여한 팀의 승리에 가까운 정도를 계산
                winning_p_w = (result[result_index][i][win_index] + 1) / 2
                winning_p_l = (result[result_index][i][lose_index] + 1) / 2
                # print('%d 경기 : 승리 팀 승리 확률 %.2f %%:' % (i, winning_p_w / (winning_p_w + winning_p_l) * 100))
                # print('%d 경기 : 패배 팀 승리 확률 %.2f %%:' % (i, winning_p_l / (winning_p_w + winning_p_l) * 100))

                # 경기 예측 요약

                # 승 패 팀 이름 할당
                win_team_name = win_info_list[win_index]
                lose_team_name = win_info_list[lose_index]
                win_team_name, etc = str(win_team_name).split('_')
                lose_team_name, etc = str(lose_team_name).split('_')

                result_df.loc[pred_index + i, 'home team'] = k_test_df.loc[pred_index + i]['home team']
                result_df.loc[pred_index + i, 'away team'] = k_test_df.loc[pred_index + i]['away team']

                result_df.loc[pred_index + i, '실제 승리 팀'] = win_team_name

                if win_team_name == k_test_df.loc[pred_index + i]['home team']:
                    result_df.loc[pred_index + i, 'home team 승리 확률'] = winning_p_w / (winning_p_w + winning_p_l) * 100
                    result_df.loc[pred_index + i, 'away team 승리 확률'] = winning_p_l / (winning_p_w + winning_p_l) * 100

                elif lose_team_name == k_test_df.loc[pred_index + i]['home team']:
                    result_df.loc[pred_index + i, 'home team 승리 확률'] = winning_p_l / (winning_p_w + winning_p_l) * 100
                    result_df.loc[pred_index + i, 'away team 승리 확률'] = winning_p_w / (winning_p_w + winning_p_l) * 100

                if winning_p_w > winning_p_l:
                    result_df.loc[pred_index + i, '예측 승리 팀'] = win_team_name
                    result_df.loc[pred_index + i, '적중 여부'] = True
                else:
                    result_df.loc[pred_index + i, '예측 승리 팀'] = lose_team_name
                    result_df.loc[pred_index + i, '적중 여부'] = False

                if winning_p_w > winning_p_l:
                    correct_pred_count += 1
            # 취소된 경기일 경우
            else:
                canceled_game_count += 1
                continue

        print(result_df.loc[pred_index: (pred_index + batch_size - 1 - canceled_game_count)])

        total_correct_pred_count += correct_pred_count
        total_canceled_game_count += canceled_game_count

        print('맞춘 경기 수 / 예측 경기 수: %d / %d' % (correct_pred_count, batch_size - canceled_game_count))
        print('적중률 : %.2f %%' % ((correct_pred_count / (batch_size - canceled_game_count)) * 100))

        # 적중률 등장 기록
        is_in = False
        for col in stoc_result.columns.values:
            if col == (str((correct_pred_count / (batch_size - canceled_game_count)) * 100) + '%'):
                stoc_result.loc[0, str((correct_pred_count / (batch_size - canceled_game_count)) * 100) + '%'] += 1
                is_in = True
                break
        if not is_in:
            stoc_result.loc[0, str((correct_pred_count / (batch_size - canceled_game_count)) * 100) + '%'] = 1

        stoc_result.loc[0, '예측 시도 일 수'] += 1

        # 예측을 시도한 경기 하나를 학습데이터에 추가
        X_train = np.concatenate([X_train, X_add_to_train], axis=0)
        y_train = np.concatenate([y_train, y_add_to_train], axis=0)

        # 다음 예측 데이터를 위한 인덱스 갱신
        pred_index += 5 * 2
        result_index += 1

        print('예측 시도 전체 경기 수 :', total_pred_count - total_canceled_game_count)
        print('맞춘 경기 수 :', total_correct_pred_count)
        print('전체 적중률 : %.2f %%' % ((total_correct_pred_count / (total_pred_count - total_canceled_game_count) * 100)))

        stoc_column_list = list(stoc_result.columns.values)
        stoc_column_list.sort()
        print(stoc_result)

    # end while prediction

    result_df = result_df.replace(np.nan, -2)
    result_df = result_df[result_df['home team'] != -2]
    result_df.reset_index(drop=True, inplace=True)
    result_df.to_csv('lstm 경기 예측 요약.csv', encoding='ms949')
    stoc_column_list = list(stoc_result.columns.values)
    stoc_column_list.sort()
    stoc_result = stoc_result.loc[:, stoc_column_list]
    stoc_result.to_csv('lstm 경기 예측율 집계.csv', encoding='ms949')