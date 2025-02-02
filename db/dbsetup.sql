SET foreign_key_checks = 0;

DROP TABLE IF EXISTS memo;
CREATE TABLE memo (
    id VARCHAR(50) NOT NULL,
    title VARCHAR(50) NOT NULL,
    content VARCHAR(1000) NOT NULL,
    created_at INT NOT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS annotation_master;
CREATE TABLE annotation_master (
    id INT NOT NULL AUTO_INCREMENT,
    label VARCHAR(50) NOT NULL,
    word VARCHAR(50) NOT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS memo_annotation;
CREATE TABLE memo_annotation (
    id INT NOT NULL AUTO_INCREMENT,
    memo_id VARCHAR(50) NOT NULL,
    annotation_id INT,
    PRIMARY KEY (id)
    -- FOREIGN KEY (memo_id) REFERENCES memo(id),
    -- FOREIGN KEY (annotation_id) REFERENCES annotation_master(id)
);

DROP TABLE IF EXISTS user;
CREATE TABLE user (
    id INT NOT NULL AUTO_INCREMENT,
    user_id VARCHAR(64) NOT NULL,
    password VARCHAR(64) NOT NULL,
    session_id VARCHAR(64),
    session_id_created_at INT,
    PRIMARY KEY (id)
    -- FOREIGN KEY (memo_id) REFERENCES memo(id),
    -- FOREIGN KEY (annotation_id) REFERENCES annotation_master(id)
);

-- INSERT INTO memo VALUES
-- (0, 'dummy', 'dummy', 1731120815);

INSERT INTO annotation_master VALUES
(1, 'programming', 'Python'),
(2, 'programming', 'JavaScript'),
(3, 'programming', 'React'),
(4, 'programming', 'Node.js'),
(5, 'programming', 'Java'),
(6, 'programming', 'C++'),
(7, 'programming', 'Ruby'),
(8, 'programming', 'Ruby On Rails'),
(9, 'programming', 'C#'),
(10, 'programming', '.NETフレームワーク'),
(11, 'programming', 'PHP'),
(12, 'database', 'MySQL'),
(13, 'database', 'PostgreSQL'),
(14, 'database', 'Microsoft SQL Server'),
(15, 'database', 'Oracle Database'),
(16, 'database', 'SQLite'),
(17, 'programming', 'プログラミング'),
(18, 'database', 'データベース'),
(19, 'webserver', 'Webサーバ'),
(20, 'webserver', 'Nginx'),
(21, 'webserver', 'Apache'),
(22, 'webserver', 'LiteSpeed'),
(23, 'webserver', 'Caddy'),
(24, 'webserver', 'Cherokee')
;

-- INSERT INTO memo_annotation VALUES
-- (1, '0001A', 1),
-- (2, '0001A', 3),
-- (3, '0001B', 2),
-- (4, '0001B', 4),
-- (5, '0001C', 1);

INSERT INTO user VALUES
(1, '0001A', '58922dc82937a44156ff0c13908b4ce4f9aa04d02d47043fad6ac71287896715', NULL, NULL);

SET foreign_key_checks = 1;
