from sys import path
import pandas as pd
import os
import datetime

import rospy


class DataGatherer:
    df_columns = ['episode', 'steps', 'reward',
                  'alpha', 'gamma', 'epsilon', 'time']
    mainDF = None
    checkpointDF = None
    start_episode_number = 0
    start_time_difference = 0

    def __init__(self, path):
        self.path = path

        if os.path.exists(self._get_file_name()):
            self.mainDF = pd.read_csv(self._get_file_name())
        else:
            self.mainDF = pd.DataFrame(columns=self.df_columns)
            with open(self._get_file_name(), 'w') as file:
                self.mainDF.to_csv(file, header=True)

        self.start_episode_number = self._get_last_episode()
        self.start_time_difference = self._get_last_time()
        self.create_checkpoint()

    def _get_file_name(self) -> str:
        return os.path.join(self.path, "logs.csv")

    def create_checkpoint(self) -> None:
        if self.checkpointDF is not None:
            self._save_checkpoint()

        self.checkpointDF = pd.DataFrame(columns=self.df_columns)

    def add_to_checkpoint(
        self,
        episode: str,
        steps: int,
        reward: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        time: datetime.timedelta
    ) -> None:
        if self.checkpointDF is None:
            self.create_checkpoint()

        self.checkpointDF = self.checkpointDF.append(
            {
                'episode': episode,
                'steps': steps,
                'reward': reward,
                'alpha': alpha,
                'gamma': gamma,
                'epsilon': epsilon,
                'time': str((time + self.start_time_difference))
            },
            ignore_index=True
        )

    def _get_last_episode(self) -> int:
        
        if len(self.mainDF.tail(1)['episode'].values) == 0:
            return 0

        return self.mainDF.tail(1)['episode'].values[0]

    def _get_last_time(self) -> datetime.time:
        
        if len(self.mainDF.tail(1)['time'].values) == 0:
            return datetime.timedelta(days = 0, hours=0, minutes=0, seconds=0)

        if "days" in self.mainDF.tail(1)['time'].values[0]:
            time = self.mainDF.tail(1)['time'].values[0].split(' ')[2].split(':')
        else:
            time = self.mainDF.tail(1)['time'].values[0].split(':')

        return datetime.timedelta(hours=int(time[0]), minutes=int(time[1]), seconds=int(time[2]))

    def _save_checkpoint(self) -> None:
        rospy.loginfo('CHECKPOINT')
        with open(self._get_file_name(), 'a') as f:
            self.checkpointDF.to_csv(f, header=f.tell() == 0)

        self.mainDF = pd.concat(
            [self.mainDF, self.checkpointDF], ignore_index=True, sort=False)
        self.checkpointDF = None


