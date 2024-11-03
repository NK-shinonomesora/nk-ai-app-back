SET foreign_key_checks = 0;

DROP TABLE IF EXISTS memo;
CREATE TABLE memo (
    id VARCHAR(50) NOT NULL,
    title VARCHAR(50) NOT NULL,
    content VARCHAR(1000) NOT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS annotation_master;
CREATE TABLE annotation_label_master (
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
    PRIMARY KEY (id),
    FOREIGN KEY (memo_id) REFERENCES memo(id),
    FOREIGN KEY (annotation_id) REFERENCES annotation_master(id)
);

INSERT INTO memo VALUES
('0001A', 'memotitle', 'memocontent'),
('0001B', 'memotitle2', 'memocontent2'),
('0001C', 'memotitle3', 'memocontent3');

INSERT INTO annotation_master VALUES
(1, 'Programming', 'C言語'),
(2, 'Programming', 'Java'),
(3, 'Database', 'MySQL')
(4, 'Database', 'Oracle Database');

INSERT INTO memo_annotation VALUES
(1, '0001A', 1),
(2, '0001A', 3),
(3, '0001B', 2),
(4, '0001B', 4),
(5, '0001C', NULL);

SET foreign_key_checks = 1;
