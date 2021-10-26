#2차원 전처리 
#크롤링 끝난 데이터를 불러와서 LSTM 2차원데이터로 가공
#훈련용 데이터하고 테스트 데이터를 나누기 직전까지 하는거
#LSTM 모델 심층 신경망 모델은 즉 LSTM이나 RNN은 심층 신경망이 일반적으로 층을 쌓고 옆으로도 쌓는다.(code로 구현하면 for문)
#시간 순으로 어떤 순서의 개념을 가질 수 있게 구성되어 있는거 :LSTM

import pandas as pd
import numpy as np
import itertools
import datetime

#일반적으로 경기를  5경기 한다면 5경기에대한 순서는 없다.
#다음 날 경기는 명백히 순서가 존재한다.
#LSTM 입력을 줄 때 이 5경기를 하나의 입력으로 데이터를 줬으면 그 다음 5경기는 다음 for문에 입력을 줘야한다. 
#즉 이 데이터를 차원을 하나 추가시킬 필요가 있다. 일반적으로  Neural net은 batch차원= row개수 몇 개씩 잘라 넣을 것인가  feature차원=column 개수 2차원의면 된다. 
#그러나 LSTM은 time step 3차원이 필요하다. 
# 코드에 일련번호를 부여하여 사전형으로 반환하는 함수-하드코딩 선수들 포지션들을 한글로 못써서 일관된 숫자로 바꿔줄 필요가 있다. 그래서 리스트를 쭉 뽑아서 종류가 11개로 치면 
# 0부터 10까지 부여한다.
def serial2dict(code_data):
    counter = 0
    result_dict = {}
    # k는 그룹화된 데이터 값, v는 그룹화된 오브젝트
    for k, v in itertools.groupby(sorted(code_data)):
        result_dict[k] = counter
        counter += 1
        # print(k, list(v))
    # print(result_dict)
    return result_dict


# 코드데이터를 직렬화 하는 함수
def serialization(raw_data, target_column):
    target_list = raw_data[target_column]
    result_dict = serial2dict(target_list)

    for key in result_dict:
        # print(key, result_dict[key])
        raw_data.loc[raw_data[target_column] == key, target_column] = result_dict[key]
    # print(raw_data[target_column])
    return raw_data

#slicing_data 날짜 데이터를 처리해야한다. 날짜를 데이터프레임으로 처리해줘야 한다.
# 날짜에 따라 잘려진 dataframe을 반환하는 함수 : 시작 날짜(이상), 끝 날짜(미만), 원본 데이터
def slicing_data(start_date, end_date, raw_data):
    print('입력으로 들어온 날짜:', start_date, end_date)
    to_drop = raw_data['날짜'].astype(int)
    # head나 tail은 series 데이터를 반환하므로 int를 통해 값만 추출
    if int(to_drop.head(1)) < start_date:
        to_drop_head = to_drop[to_drop < start_date]
        start_date_index = to_drop_head.index.values[-1] + 1
        if start_date_index < raw_data.loc[:, '날짜'].tail(1).index.values[0]:
            start_date = raw_data.loc[start_date_index, '날짜']
        else:
            print('잘린 후 남은 데이터가 없습니다.')
            start_date = raw_data.loc[:, '날짜'].tail(1).values[0]
    else:
        start_date = int(to_drop.head(1))
        start_date_index = to_drop.index.values[0]

    if end_date < int(to_drop.tail(1)):
        to_drop_tail = to_drop[end_date < to_drop]
        end_date_index = to_drop_tail.index.values[0]
        end_date = raw_data.loc[end_date_index, '날짜']
    else:
        end_date = int(to_drop.tail(1))
        end_date_index = to_drop.tail(1).index.values[0] + 1
    print('drop index 생성 완료', start_date, '(', start_date_index, '), ', end_date, '(', end_date_index, ')')

    if start_date_index < end_date_index:
        raw_data = raw_data[start_date_index:]
        print('시작 날짜 이전 데이터 자르기 완료')
        raw_data = raw_data[:end_date_index]
        print('종료 날짜 이후 데이터 자르기 완료')
    else:
        # raw_data.drop(raw_data.index, inplace=True)
        raw_data = raw_data.iloc[0:0]

    return raw_data

#월 추가 
# datetime의 개월 단위 덧셈 연산
def add_months(date, months):
    month = date.month + months - 1
    year = date.year + int(month / 12)
    month = month % 12 + 1

    return datetime.date(year, month, date.day)

#정규화
# target column들을 정규화하는 함수
def normalize_data(target_columns, raw_data):
    transformed_data = pd.DataFrame(data=raw_data)
    for col in target_columns:
        # print(raw_data[col])
        col_mean = np.mean(raw_data[col])
        # print(col_mean)
        col_std = np.std(raw_data[col])
        # print(col_std)
        if col_std != 0:
            # print(raw_data[col].max())
            transformed_data[col] = (raw_data[col] - col_mean) / col_std
            # print(col, '의 max :', transformed_data[col].max())
        else:
            # print(col, '의 표준편차는', col_std)
            transformed_data = transformed_date.drop(col, axis='columns')

    return transformed_data

#하드코딩데어있는데 2019->2018로 바꾸면 된다.
if __name__ == '__main__':
    game_data_url = '2019_KBO_경기 데이터.csv'

    game_data = pd.read_csv(game_data_url)

    # 날짜별 오름차순 정렬
    game_data = game_data.sort_values('날짜', ascending=True).reset_index(drop=True)

    # 게임 데이터 처리
    # 공백을 0으로 채움
    game_data.replace(np.nan, 0, inplace=True)
    # 날짜정보가 없는 데이터는 제거
    game_data = game_data[game_data['날짜'] != 0]
    game_data.drop_duplicates(inplace=True)#중복된 경우가 있을 수 있다. 여러번 경기를 읽어오면서 오류  테이블을 merge할 때 중복을 시킨채로 merge 그래서 다시불러올 때 혹시 있으면 duplicate drop 해주고
    game_data.reset_index(drop=True, inplace=True)#index 세팅 다시 세팅 해준다.

    # 추가 투입 선수 여부 열 생성
    game_data.loc[:, 'only_starter'] = 0 #컬럼 onlystarter라고 starter만 있는 경기인지 starter와 교체선수가 있는 경기인지 구분하는 거 생성
    # 선발선수만으로 이루어진 데이터 생성
    game_data_with_starter = game_data.replace(0.5, 0)
    # 추가 투입 선수 여부 열 갱신
    game_data_with_starter.loc[:, 'only_starter'] = 1
    # 선발 선수만으로 이루어진 데이터의 경우 모든 점수 데이터 0 으로 변경  
    game_data_with_starter.loc[:, 'home_score'] = 0
    game_data_with_starter.loc[:, 'away_score'] = 0
    # 병합
    game_data = pd.concat((game_data, game_data_with_starter), axis=0)
    game_data.reset_index(drop=True, inplace=True)

    # 날짜별 오름차순 정렬. 이후, 시간 순서 정리. only_starter == 1 인 데이터가 우선 발생(only_starter 내림차순 정렬)
    game_data = game_data.sort_values(['날짜', 'only_starter'], ascending=[True, False])

    # 팀 list 생성
    team_list = game_data['home team'].unique()

    # 처리된 팀별 데이터를 담을 list
    transformed_merged_df_list = []
    k_transformed_merged_df_list = []

    # 목표 팀을 기준으로 경기 데이터 생성
    for target_team in team_list:
        target_game_data = game_data[(game_data['home team'] == target_team) | (game_data['away team'] == target_team)]
        target_game_data = target_game_data.reset_index(drop=True)

        merged_game_data = target_game_data

        # 한글 이름을 유지할 게임 데이터
        # call by reference가 되므로 k_merged_game_data = game_data 이렇게 할당하면 안 된다
        k_merged_game_data = target_game_data.copy()

        # win 열 생성 #레이블 생성하는 부분 away가 이겼냐 졌냐
        merged_game_data.loc[
            (merged_game_data['home team'] == target_team) & (merged_game_data['away result'] == 0), str(
                target_team) + '_win'] = 1
        merged_game_data.loc[
            (merged_game_data['home team'] == target_team) & (merged_game_data['away result'] == 1), str(
                target_team) + '_win'] = 0
        merged_game_data.loc[
            (merged_game_data['away team'] == target_team) & (merged_game_data['away result'] == 0), str(
                target_team) + '_win'] = 0
        merged_game_data.loc[
            (merged_game_data['away team'] == target_team) & (merged_game_data['away result'] == 1), str(
                target_team) + '_win'] = 1

        k_merged_game_data.loc[
            (k_merged_game_data['home team'] == target_team) & (k_merged_game_data['away result'] == 0), str(
                target_team) + '_win'] = 1
        k_merged_game_data.loc[
            (k_merged_game_data['home team'] == target_team) & (k_merged_game_data['away result'] == 1), str(
                target_team) + '_win'] = 0
        k_merged_game_data.loc[
            (k_merged_game_data['away team'] == target_team) & (k_merged_game_data['away result'] == 0), str(
                target_team) + '_win'] = 0
        k_merged_game_data.loc[
            (k_merged_game_data['away team'] == target_team) & (k_merged_game_data['away result'] == 1), str(
                target_team) + '_win'] = 1

        # away result 제거 #레이블을 따로 생성했으니까 awayresult를 제거
        merged_game_data.drop('away result', axis=1, inplace=True)
        k_merged_game_data.drop('away result', axis=1, inplace=True)

        # 전체 승률 계산 #전체승률이 있으면 상대팀과의 승률 관계도 있다. ex)한화와 LG한 경기 그것만 통계 밑에부분(상대 승률) 낸거 한화가 모든 경기를 통계낸거는 위에부분
        for index in merged_game_data.index:  
            # 아직 한 경기도 진행되지 않았을 때
            if index == 0:
                # 50% 할당
                merged_game_data.loc[index, str(target_team) + '_overall_odds'] = 0.5
                k_merged_game_data.loc[index, str(target_team) + '_overall_odds'] = 0.5
            # 한 경기라도 진행됐다면
            else:
                # 이전 경기까지의 승률 할당
                merged_game_data.loc[index, str(target_team) + '_overall_odds'] = sum(
                    merged_game_data.iloc[:index][str(target_team) + '_win']) / len(merged_game_data.iloc[:index])
                k_merged_game_data.loc[index, str(target_team) + '_overall_odds'] = sum(
                    k_merged_game_data.iloc[:index][str(target_team) + '_win']) / len(k_merged_game_data.iloc[:index])

        # team_list = merged_game_data['home team'].unique()

        # target team 의 index를 찾아 target team 이 제거된 team list 생성
        team_list_no_target = np.delete(team_list.copy(), np.where(team_list == target_team))
        # 상대팀과의 승률 계산
        for opponent in team_list_no_target:
            game_data_with_opponent = merged_game_data[
                (merged_game_data['home team'] == opponent) | (merged_game_data['away team'] == opponent)]
            o_index = 0
            for index in game_data_with_opponent.index:
                # 아직 한 경기도 진행되지 않았을 때
                if o_index == 0:
                    # 50% 할당
                    merged_game_data.loc[index, str(target_team) + '_odds_per_team'] = 0.5
                    k_merged_game_data.loc[index, str(target_team) + '_odds_per_team'] = 0.5
                # 한 경기라도 진행됐다면
                else:
                    # 이전 경기까지의 승률 할당 -> 수식 오류 있네! (21.05.23 발견)
                    merged_game_data.loc[index, str(target_team) + '_odds_per_team'] = sum(
                        game_data_with_opponent.iloc[:o_index][str(target_team) + '_win'] /
                        len(game_data_with_opponent.iloc[:o_index][str(target_team) + '_win']))
                    k_merged_game_data.loc[index, str(target_team) + '_odds_per_team'] = sum(
                        game_data_with_opponent.iloc[:o_index][str(target_team) + '_win'] /
                        len(game_data_with_opponent.iloc[:o_index][str(target_team) + '_win']))
                o_index += 1

        # 생성된 열의 위치 이동
        # 선수 테이블과 분리하는 작업이 이후에 존재하므로 win 열 보다 앞으로 이동
        column_list = list(merged_game_data.columns.values)

        # win 열부터 뒤로 밀림
        column_list.insert(merged_game_data.columns.get_loc(str(target_team) + '_win'),
                           column_list.pop(column_list.index(str(target_team) + '_odds_per_team')))
        column_list.insert(merged_game_data.columns.get_loc(str(target_team) + '_win'),
                           column_list.pop(column_list.index(str(target_team) + '_overall_odds')))
        merged_game_data = merged_game_data.loc[:, column_list]
        k_merged_game_data = k_merged_game_data.loc[:, column_list]
        print(str(target_team) + ' - 전체 승률, 상대 팀 별 승률 계산 완료')

        transformed_merged_df_list.append(merged_game_data)
        k_transformed_merged_df_list.append(k_merged_game_data)
    # end for team_list

    # 처리된 팀별 경기 데이터를 병합 #테이블 2개를 만들었으니까 merge 병합
    # 병합을 위해 모든 팀별 경기 데이터가 공통으로 갖는 column list 생성 (날짜 ~ only_starter)
    columns_to_merge = list(transformed_merged_df_list[0].columns.values)
    columns_to_merge = columns_to_merge[:transformed_merged_df_list[0].columns.get_loc('only_starter') + 1]

    full_game_merged_data = pd.merge(transformed_merged_df_list[0], transformed_merged_df_list[1], how='outer',
                                     on=columns_to_merge)
    k_full_game_merged_data = pd.merge(k_transformed_merged_df_list[0], k_transformed_merged_df_list[1], how='outer',
                                       on=columns_to_merge)

    for transformed_merged_df, k_transformed_merged_df in zip(transformed_merged_df_list[2:],
                                                              k_transformed_merged_df_list[2:]):
        full_game_merged_data = pd.merge(full_game_merged_data, transformed_merged_df, how='outer', on=columns_to_merge)
        k_full_game_merged_data = pd.merge(k_full_game_merged_data, k_transformed_merged_df, how='outer',
                                           on=columns_to_merge)

    # 공백 처리 #병합하면서 생긴 공백처리
    full_game_merged_data.replace(np.nan, -1, inplace=True)
    k_full_game_merged_data.replace(np.nan, -1, inplace=True)
    print('처리된 팀 데이터 병합 완료')

    # 날짜순, 경기 전 선발선수데이터가 우선되는 순으로 정렬 #오름차순으로 정렬을 날짜순으로 정리하되 경기전 선발 선수가 먼저 생성됨 선발선수경기, 선발교체경기 두 경우가 있는데 교차하면서 생성된다. 
    full_game_merged_data = full_game_merged_data.sort_values(['날짜', 'only_starter'], ascending=[True, False])# 이거를 순서정렬한거
    k_full_game_merged_data = k_full_game_merged_data.sort_values(['날짜', 'only_starter'], ascending=[True, False])

    full_game_merged_data.reset_index(drop=True, inplace=True)
    k_full_game_merged_data.reset_index(drop=True, inplace=True)
    print('데이터 정렬 완료')

    # 팀별 전체 승률과 팀별 팀 승률 갱신
    for team in team_list:
        for index, overall_odds, odds_per_team in zip(range(0, len(full_game_merged_data)),
                                                      full_game_merged_data[str(team) + '_overall_odds'],
                                                      full_game_merged_data[str(team) + '_odds_per_team']):
            latest_overall_odds = 0.5
            latest_odds_per_team = 0.5
            if overall_odds != -1:
                latest_overall_odds = overall_odds
            else:
                full_game_merged_data.loc[index, str(team) + '_overall_odds'] = latest_overall_odds

            if odds_per_team != -1:
                latest_odds_per_team = odds_per_team
            else:
                full_game_merged_data.loc[index, str(team) + '_odds_per_team'] = latest_odds_per_team
    print('처리된 데이터 팀별 전체 승률, 팀별 팀 승률 갱신 완료')

    # 날짜, 홈팀, 원정팀까지만 담긴 df와 이후 정보가 담긴 df 잠시 분리
    temp_team_df, temp_etc_df = full_game_merged_data.loc[:, :'away team'], full_game_merged_data.loc[:, '장소':]

    # home, away 팀을 one hot 으로 변경
    for team in team_list:
        temp_team_df.loc[temp_team_df['home team'] == team, team] = 1
        temp_team_df.loc[temp_team_df['away team'] == team, team] = 1

    temp_team_df.replace(np.nan, 0, inplace=True)
    # 변경된 데이터프레임을 병합
    full_game_merged_data = pd.concat((temp_team_df, temp_etc_df), axis=1)
    full_game_merged_data.drop('home team', axis=1, inplace=True)
    full_game_merged_data.drop('away team', axis=1, inplace=True)
    print('팀 one hot 으로 변경 완료')

    place_list = full_game_merged_data['장소'].unique()

    # 경기 구장 one hot 으로 변경 # 한글로 되어있는걸 one hot으로 컬럼을 나눠준다.
    b_place_df, a_place_df = full_game_merged_data.loc[:, :'장소'], full_game_merged_data.loc[:, '장소':]
    a_place_df.drop('장소', axis=1, inplace=True)
    for place in place_list:
        b_place_df.loc[b_place_df['장소'] == place, place] = 1
    b_place_df.drop('장소', axis=1, inplace=True)
    b_place_df.replace(np.nan, 0, inplace=True)
    full_game_merged_data = pd.concat((b_place_df, a_place_df), axis=1)
    print('경기 구장 one hot 으로 변경 완료')

    # 경기 분류
    game_list = full_game_merged_data['경기 분류'].unique()
    for game in game_list:
        n, temp = str(game).split('차')
        full_game_merged_data.loc[full_game_merged_data['경기 분류'] == game, '경기 분류'] = n
    k_full_game_merged_data['경기 분류'] = full_game_merged_data['경기 분류']
    print('경기 분류 처리 완료')

    # 동일한 값만 갖는 열 제거( 11111110000000공백이런거 제거)
    dup_column_list = full_game_merged_data.columns.values
    dup_column_list = dup_column_list.tolist()

    # win, h_score, a_score, odds_per_team, overall_odds 제외
    for target_team in team_list:
        dup_column_list.remove(str(target_team) + '_win')
        dup_column_list.remove(str(target_team) + '_odds_per_team')
        dup_column_list.remove(str(target_team) + '_overall_odds')
    dup_column_list.remove('home_score')
    dup_column_list.remove('away_score')
    # 날짜 제외(LSTM에 날짜 컬럼은 안들어 간다.)
    dup_column_list.remove('날짜')
    # 구장 제외( 제거해 줄 리스트중 구장은 있어야하니까 중복된 구장을 갖고올떄도 있다. 그런경우는 구장을 빼고 검사해라)
    for place in place_list:
        dup_column_list.remove(place)

    for col in dup_column_list:
        dif_list = full_game_merged_data[col].unique()
        dif_n = full_game_merged_data[col].nunique()
        # 동일한 값만 존재하는 경우
        if (dif_n == 1):
            if col not in team_list:
                full_game_merged_data = full_game_merged_data.drop(col, axis=1)
                k_full_game_merged_data = k_full_game_merged_data.drop(col, axis=1)
            print(col, '열 제거. 유일 값 :', dif_list[0])
        else:
            pass
    print('동일 열 제거 완료')

    # 학습 이터와 평가 데이터 분리
    # 날짜 데이터 리스트 추출
    date_list = full_game_merged_data['날짜'].unique()
    # 요일은 숫자로 변경
    date_str_to_int_dic = {
        '일': 0,
        '월': 1,
        '화': 2,
        '수': 3,
        '목': 4,
        '금': 5,
        '토': 6,
    }

    for date in date_list:
        # 날짜 데이터의 요일 분리
        transformed_date, day_of_the_week = str(date).split('(')
        day_of_the_week, etc = str(day_of_the_week).split(')')
        # 날짜 데이터 사이의 . 제거 & 요일 정보 추가
        full_game_merged_data.loc[full_game_merged_data['날짜'] == date, '요일'] = date_str_to_int_dic[day_of_the_week]
        full_game_merged_data.loc[full_game_merged_data['날짜'] == date, '날짜'] = transformed_date.replace('.', '')
        k_full_game_merged_data.loc[k_full_game_merged_data['날짜'] == date, '요일'] = date_str_to_int_dic[day_of_the_week]
        k_full_game_merged_data.loc[k_full_game_merged_data['날짜'] == date, '날짜'] = transformed_date.replace('.', '')

    # 요일 열을 두 번째 열로 위치 변경
    column_list = list(full_game_merged_data.columns.values)
    k_column_list = list(k_full_game_merged_data.columns.values)
    column_list.insert(1, column_list.pop(column_list.index('요일')))
    k_column_list.insert(1, k_column_list.pop(k_column_list.index('요일')))

    full_game_merged_data = full_game_merged_data.loc[:, column_list]
    k_full_game_merged_data = k_full_game_merged_data.loc[:, k_column_list]
    print('날짜, 요일 정보 분리 완료')

    # 날짜별 오름차순 정렬. 이후, 시간 순서 정리. only_starter == 1 인 데이터가 우선 발생(only_starter 내림차순 정렬)
    full_game_merged_data = full_game_merged_data.sort_values(['날짜', 'only_starter'], ascending=[True, False])
    k_full_game_merged_data = k_full_game_merged_data.sort_values(['날짜', 'only_starter'], ascending=[True, False])
    full_game_merged_data.reset_index(drop=True, inplace=True)
    k_full_game_merged_data.reset_index(drop=True, inplace=True)

    # 데이터의 처음 날짜를 학습 데이터의 시작 날짜로 지정
    start_date_train = datetime.datetime.strptime(str(full_game_merged_data['날짜'][0]), "%Y%m%d")
    # 학습 데이터의 처음 날짜 임의 지정
    # start_date_train = datetime.datetime.strptime('20190601', '%Y%m%d')
    # n 개월치의 데이터를 학습데이터로 지정
    end_date_train = add_months(start_date_train, 2)
    # 학습 데이터의 마지막 날짜 임의 지정
    end_date_train = datetime.datetime.strptime('20190531', '%Y%m%d')

    # 추출할 평가 데이터 기간 설정, 평가 데이터 기간 이후 n 개월치
    start_date_test = end_date_train
    # 학습 데이터 마지막 날의 다음 날부터 평가 데이터로 설정
    start_date_test = start_date_test + datetime.timedelta(days=1)
    # 평가 데이터의 시작 날짜 임의 지정
    # start_date_test = datetime.datetime.strptime('20190801', '%Y%m%d')·
    # end_date_test = add_months(start_date_test, 99)
    # 평가 데이터의 종료 날짜 임의 지정
    end_date_test = datetime.datetime.strptime('20191001', '%Y%m%d')

    # 만들어낸 날짜를 크기 비교가 가능한 int 값으로 변경
    start_date_train = int(datetime.date.strftime(start_date_train, "%Y%m%d"))
    end_date_train = int(datetime.date.strftime(end_date_train, "%Y%m%d"))

    start_date_test = int(datetime.date.strftime(start_date_test, "%Y%m%d"))
    end_date_test = int(datetime.date.strftime(end_date_test, "%Y%m%d"))

    # 학습 기간, 평가 기간을 포함한 데이터로 자르기
    full_game_merged_data = slicing_data(start_date_train, end_date_test, full_game_merged_data)
    full_game_merged_data.reset_index(drop=True, inplace=True)
    k_full_game_merged_data = slicing_data(start_date_train, end_date_test, k_full_game_merged_data)
    k_full_game_merged_data.reset_index(drop=True, inplace=True)
    print('데이터 기간 분리 완료')

    # 기간 분리 이후 동일한 값만 갖는 열 제거
    dup_column_list = full_game_merged_data.columns.values
    dup_column_list = dup_column_list.tolist()

    # win, h_score, a_score, odds_per_team, overall_odds 제외
    for target_team in team_list:
        dup_column_list.remove(str(target_team) + '_win')
        dup_column_list.remove(str(target_team) + '_odds_per_team')
        dup_column_list.remove(str(target_team) + '_overall_odds')
    dup_column_list.remove('home_score')
    dup_column_list.remove('away_score')
    # 날짜 제외
    dup_column_list.remove('날짜')
    # 구장 제외
    for place in place_list:
        dup_column_list.remove(place)

    for col in dup_column_list:
        dif_list = full_game_merged_data[col].unique()
        # 동일한 값만 존재하는 경우
        if (len(dif_list) == 1):
            full_game_merged_data = full_game_merged_data.drop(col, axis=1)
            k_full_game_merged_data = k_full_game_merged_data.drop(col, axis=1)
            print(col, '열 제거. 유일 값 :', dif_list[0])

    print('기간 분리 이후 동일 열 제거 완료')

    # 날짜를 기준으로 학습 데이터와 평가 데이터로 분리
    lstm_train_data = slicing_data(start_date_train, end_date_train, full_game_merged_data)
    lstm_test_data = slicing_data(start_date_test, end_date_test, full_game_merged_data)

    k_train_data = slicing_data(start_date_train, end_date_train, k_full_game_merged_data)
    k_test_data = slicing_data(start_date_test, end_date_test, k_full_game_merged_data)

    # 날짜 정보 제거
    # lstm_train_data.drop('날짜', axis='columns', inplace=True)
    # lstm_test_data.drop('날짜', axis='columns', inplace=True)

    print('학습 데이터, 평가 데이터 분리 완료')

    """
    column_list = list(lstm_train_data.columns.values)
    del column_list[column_list.index('win')]
    del column_list[column_list.index('odds_per_team')]
    del column_list[column_list.index('overall_odds')]
    del column_list[column_list.index('경기 분류')]
    for col in team_list:
        column_list.remove(col)
    for col in place_list:
        column_list.remove(col)

    # 열 검사. test에는 값이 있지만 train에는 값이 없는 데이터가 있는가?
    # 또는 train에만 있고 test에는 없는 학습할 필요 없는 데이터가 있는가?
    no_train_c = 0
    no_test_c = 0
    total_train_c = len(lstm_train_data)
    total_test_c = len(lstm_test_data)

    for col in column_list:
        train_val = lstm_train_data[col].nunique()
        test_val = lstm_test_data[col].nunique()
        # 테스트에는 있지만 학습에는 없는 경우는 테스트 데이터에서 제외
        if (train_val == 1) & (test_val == 2):
            print(col, ': train 경기 수', len(lstm_train_data[lstm_train_data[col] == 1]), '/', len(lstm_train_data),
                  'test 경기 수', len(lstm_test_data[lstm_test_data[col] == 1]), '/', len(lstm_test_data))
            print('test 전체 경기', len(lstm_test_data), '회 중', col, '이(가) 출전한 경기 수 :',
                  len(lstm_test_data[lstm_test_data[col] == 1]))
            no_train_c += len(lstm_test_data[lstm_test_data[col] == 1])
            # lstm_test_data.drop(lstm_test_data[lstm_test_data[col] == 1].index, inplace=True)
            # k_test_data.drop(k_test_data[k_test_data[col] == 1].index, inplace=True)

        # 학습에는 있지만 테스트에는 없을 경우 학습할 필요 없으니 제외.. 하면 안돼
        if (train_val == 2) & (test_val == 1):
            test_data_flag = True
            for col2 in column_list:
                # col 은 테스트에 없지만, col 이 참여한 경기가 test에 존재하는 경기에 참여한 선수(col2)와 함께였다면 제거 행에서 제외
                if (len(lstm_test_data[col2].unique()) == 2) & len(lstm_train_data[(lstm_train_data[col] == 1) & (lstm_train_data[col2] == 1)]) > 0:
                    test_data_flag = False
                    #print(col, '이(가) 참여한 경기는', col2, '와(과) test 데이터의 경기에서 함께하므로 제거대상에서 제외')
                    break
            # col이 참여한 경기는 test에 존재하는 선수 그 누구와도 함께한 적이 없는 경기
            if test_data_flag:
                print(col, ': train 경기 수', len(lstm_train_data[lstm_train_data[col] == 1]), '/', len(lstm_train_data),
                      'test 경기 수', len(lstm_test_data[lstm_test_data[col] == 1]), '/', len(lstm_test_data))
                print('전체 경기 수', len(lstm_train_data), '중', col, '이(가) train에서만 출전한 경기 수 :', len(lstm_train_data[lstm_train_data[col] == 1]))
                no_test_c += len(lstm_train_data[lstm_train_data[col] == 1])
                lstm_train_data.drop(lstm_train_data[lstm_train_data[col] == 1].index, inplace=True)

    print('test 전체', total_test_c, '건 중, 학습에는 없는 선수가 참여한 경기 데이터', no_train_c, '건 제거 완료')
    print('train 전체', total_train_c, '건 중, 테스트에는 없는 선수가 참여한 경기 데이터', no_test_c, '건 제거 완료')
    """

    # 제거된 데이터로 인해 필요없어진 동일한 값만 갖는 열 제거
    dup_column_list = lstm_train_data.columns.values
    dup_column_list = dup_column_list.tolist()

    # win, h_score, a_score, odds_per_team, overall_odds 제외
    for target_team in team_list:
        dup_column_list.remove(str(target_team) + '_win')
        dup_column_list.remove(str(target_team) + '_odds_per_team')
        dup_column_list.remove(str(target_team) + '_overall_odds')
    dup_column_list.remove('home_score')
    dup_column_list.remove('away_score')
    # 구장 제외
    for place in place_list:
        dup_column_list.remove(place)

    for col in dup_column_list:
        dif_list1 = lstm_train_data[col].unique()
        dif_list2 = lstm_test_data[col].unique()
        # 동일한 값만 존재하는 경우
        if (len(dif_list1) == 1) & (len(dif_list2) == 1):
            lstm_train_data = lstm_train_data.drop(col, axis=1)
            k_train_data = k_train_data.drop(col, axis=1)
            lstm_test_data = lstm_test_data.drop(col, axis=1)
            k_test_data = k_test_data.drop(col, axis=1)
            print(col, '열 제거. 유일 값 :', dif_list1[0], dif_list2[0])

    print('열 검사 이후 동일 열 제거 완료')

    # 테스트셋의 추가 선수 데이터 제외 (필요하다면)
    # lstm_test_data.replace(0.5, 0, inplace=True)
    # lstm_test_data.replace(-0.5, 0, inplace=True)
    # k_test_data.replace(0.5, 0, inplace=True)
    # k_test_data.replace(-0.5, 0, inplace=True)
    print('테스트 데이터 추가 선수 데이터 제외')

    # 모든 열의 이름 변경(b0~bn)
    lstm_column_list = list(lstm_train_data.columns.values)
    for target_team in team_list:
        lstm_column_list.remove(str(target_team) + '_win')
        # lstm_column_list.remove(str(target_team) + '_odds_per_team')
        # lstm_column_list.remove(str(target_team) + '_overall_odds')

    # 열들의 이름을 b0 ~ bn으로 변환
    index = 0
    lstm_diction = {}

    for col in lstm_column_list:
        lstm_diction[col] = 'b' + str(index)

        index += 1

    index = 0
    for target_team in team_list:
        lstm_diction[str(target_team) + '_win'] = 'win_' + str(index)
        index += 1

    lstm_train_data = lstm_train_data.rename(index=str, columns=lstm_diction)
    lstm_test_data = lstm_test_data.rename(index=str, columns=lstm_diction)

    print('열 이름 변경 완료')

    """
    # label 들의 위치를 맨 뒤로 변경
    lstm_column_list = list(lstm_train_data.columns.values)
    k_column_list = list(k_test_data.columns.values)

    #lstm_column_list.insert(index, lstm_column_list.pop(lstm_column_list.index('odds_per_team')))
    #lstm_column_list.insert(index, lstm_column_list.pop(lstm_column_list.index('overall_odds')))
    lstm_column_list.insert(len(lstm_column_list), lstm_column_list.pop(lstm_column_list.index('win')))

    #k_column_list.insert(index, k_column_list.pop(k_column_list.index('odds_per_team')))
    #k_column_list.insert(index, k_column_list.pop(k_column_list.index('overall_odds')))
    k_column_list.insert(len(k_column_list), k_column_list.pop(k_column_list.index('win')))

    lstm_train_data = lstm_train_data.loc[:, lstm_column_list]
    lstm_test_data = lstm_test_data.loc[:, lstm_column_list]
    k_train_data = k_train_data.loc[:, k_column_list]
    k_test_data = k_test_data.loc[:, k_column_list]
    print('label 들의 위치 변경 완료')
    """

    lstm_column_list = list(lstm_train_data.columns.values)
    k_column_list = list(k_test_data.columns.values)

    lstm_train_data = lstm_train_data.set_index(lstm_column_list[0])
    lstm_test_data = lstm_test_data.set_index(lstm_column_list[0])
    k_train_data = k_train_data.set_index(k_column_list[0])
    k_test_data = k_test_data.set_index(k_column_list[0])
    print('첫 번째 열을 index로 지정 완료')

    lstm_train_data.to_csv('transformed_lstm_train_data.csv', encoding='utf-8')
    lstm_test_data.to_csv('transformed_lstm_test_data.csv', encoding='utf-8')
    k_train_data.to_csv('transformed_lstm_train_data_k.csv', encoding='utf-8-sig')
    k_test_data.to_csv('transformed_lstm_test_data_k.csv', encoding='utf-8-sig')
    print('csv 파일 생성 완료')