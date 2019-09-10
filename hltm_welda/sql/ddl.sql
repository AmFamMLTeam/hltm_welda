CREATE TABLE IF NOT EXISTS save (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  name TEXT,
  K INTEGER,
  alpha_init REAL,
  eta_init REAL,
  random_state INTEGER
);


CREATE TABLE IF NOT EXISTS token_maps (
  -- FOREIGN KEY (save_id) REFERENCES save (id),
  save_id INTEGER,
  d INTEGER,
  w INTEGER,
  w_cp INTEGER,
  t INTEGER
);


CREATE TABLE IF NOT EXISTS vocabulary (
  -- FOREIGN KEY (save_id) REFERENCES save(id),
  save_id INTEGER,
  word_index INTEGER,
  word TEXT
);



CREATE TABLE IF NOT EXISTS alpha (
  -- FOREIGN KEY (save_id) REFERENCES save(id),
  save_id INTEGER,
  alpha REAL
);


CREATE TABLE IF NOT EXISTS eta (
  -- FOREIGN KEY (save_id) REFERENCES save(id),
  save_id INTEGER,
  eta REAL
);



CREATE TABLE IF NOT EXISTS topic_names (
  -- FOREIGN KEY (save_id) REFERENCES save(id),
  save_id INTEGER,
  topic_id INTEGER,
  topic_name TEXT
);
