import numpy as np
import pandas as pd
import sqlite3
import json


class Database(object):
    def __init__(self):
        self._DATABASE = 'hltm_welda/sql/lda.db'

    @property
    def DATABASE(self):
        return self._DATABASE


    def create_db_conn(self):
        return sqlite3.connect(self.DATABASE)


    def persist_token_maps_by_save_id(
        self,
        save_id,
        token_maps
    ):
        conn = self.create_db_conn()
        cur = conn.cursor()

        sql = '''
        INSERT INTO token_maps
        (
            save_id,
            d,
            w,
            w_cp,
            t
        )
        VALUES
        (
            ?,
            ?,
            ?,
            ?,
            ?
        );
        '''
        cur.executemany(
            sql,
            token_maps
        )

        conn.commit()
        conn.close()



    def persist_vocab_by_save_id(
        self,
        save_id,
        words
    ):
        conn = self.create_db_conn()
        cur = conn.cursor()

        sql = '''
        INSERT INTO vocabulary
        (
            save_id,
            word_index,
            word
        )
        VALUES
        (
            ?,
            ?,
            ?

        );
        '''

        cur.executemany(
            sql,
            words
        )
        conn.commit()
        conn.close()


    def persist_alpha_by_save_id(
        self,
        save_id,
        alpha
    ):
        conn = self.create_db_conn()
        cur = conn.cursor()

        sql = '''
        INSERT INTO alpha
        (
            save_id,
            alpha
        )
        VALUES
        (
            ?,
            ?
        );
        '''

        cur.executemany(
            sql,
            alpha
        )
        conn.commit()
        conn.close()


    def persist_eta_by_save_id(
        self,
        save_id,
        eta
    ):
        conn = self.create_db_conn()
        cur = conn.cursor()

        sql = '''
        INSERT INTO eta
        (
            save_id,
            eta
        )
        VALUES
        (
            ?,
            ?
        );
        '''

        cur.executemany(
            sql,
            eta
        )
        conn.commit()
        conn.close()


    def persist_topic_names_by_save_id(
        self,
        save_id,
        topic_names
    ):
        conn = self.create_db_conn()
        cur = conn.cursor()

        sql = '''
        INSERT INTO topic_names
        (
            save_id,
            topic_id,
            topic_name
        )
        VALUES
        (
            ?,
            ?,
            ?
        );
        '''

        cur.executemany(
            sql,
            topic_names
        )

        conn.commit()
        conn.close()


    def create_model_record(
        self,
        save_name,
        model
    ):
        conn = self.create_db_conn()
        cur = conn.cursor()

        sql = '''
        INSERT INTO save
        (
            name,
            K,
            alpha_init,
            eta_init,
            random_state
        )
        VALUES
        (
            ?,
            ?,
            ?,
            ?,
            ?,
            ?
        );
        '''

        cur.execute(
            sql,
            (
                save_name,
                model.K,
                model.alpha_init,
                model.eta_init,
                model.random_state
            )
        )
        last_row_id = cur.lastrowid
        conn.commit()
        conn.close()

        return last_row_id



    def get_save_names(self):
        conn = self.create_db_conn()
        cur = conn.cursor()

        cur.execute('''SELECT name FROM save ORDER BY timestamp DESC;''')

        names = cur.fetchall()
        conn.commit()
        names = [name[0] for name in names]
        names = list(set(names))
        conn.close()

        return names


    def get_token_maps_by_save_id(
        self,
        save_id
    ):
        conn = self.create_db_conn()
        token_maps_df = pd.read_sql_query(
            sql='''
            SELECT
            d,
            w,
            w_cp,
            t
            FROM token_maps
            WHERE save_id = ?
            ''',
            con=conn,
            params=[save_id]
        )

        conn.close()

        return token_maps_df


    def get_vocabulary_by_save_id(
        self,
        save_id
    ):
        conn = self.create_db_conn()
        vocab_dict = dict(pd.read_sql_query(
            sql='''
            SELECT
            word,
            word_index
            FROM vocabulary
            WHERE save_id = ?
            ''',
            con=conn,
            params=[save_id]
        ).values.tolist())

        conn.close()

        return vocab_dict


    def get_alpha_by_save_id(
        self,
        save_id
    ):
        conn = self.create_db_conn()
        alpha_stacked = pd.read_sql_query(
            sql='''
            SELECT
            alpha
            FROM alpha
            WHERE save_id = ?
            ''',
            con=conn,
            params=[save_id]
        ).values

        conn.close()

        return alpha_stacked


    def get_eta_by_save_id(
        self,
        save_id
    ):
        conn = self.create_db_conn()
        eta_stacked = pd.read_sql_query(
            sql='''
            SELECT
            eta
            FROM eta
            WHERE save_id = ?
            ''',
            con=conn,
            params=[save_id]
        ).values

        conn.close()

        return eta_stacked


    def get_topic_names_by_save_id(
        self,
        save_id
    ):
        conn = self.create_db_conn()
        topic_names_df = pd.read_sql_query(
            sql='''
            SELECT
            topic_id,
            topic_name
            FROM topic_names
            WHERE save_id = ?
            ''',
            con=conn,
            params=[save_id]
        )

        topic_names_records = topic_names_df.to_dict(orient='records')
        topic_names_dict = {
            d['topic_id']: d['topic_name']
            for d in topic_names_records
        }

        conn.close()

        return topic_names_dict


    def get_save_record_by_name(
        self,
        save_name
    ):
        conn = self.create_db_conn()

        save_dict = pd.read_sql_query(
            sql='''
            SELECT
            id,
            timestamp,
            name,
            K,
            alpha_init,
            eta_init,
            random_state
            FROM save
            WHERE id = (
                SELECT MAX(id) FROM save
                WHERE name = ?
            )
            LIMIT 1;
            ''',
            con=conn,
            params=[save_name]
        ).to_dict(orient='records')[0]

        conn.close()

        return save_dict
