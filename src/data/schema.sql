CREATE DATABASE IF NOT EXISTS hr_db;
USE hr_db;

-- Basic candidate info (no experience or education details here anymore)
CREATE TABLE IF NOT EXISTS candidate_profiles (
    candidate_id INT PRIMARY KEY,
    birthdate DATE,
    firstname VARCHAR(100),
    lastname VARCHAR(100)
);

-- Job experiences (multiple per candidate)
CREATE TABLE IF NOT EXISTS candidate_experiences (
    exp_id INT AUTO_INCREMENT PRIMARY KEY,
    candidate_id INT,
    company VARCHAR(255),
    company_industry VARCHAR(255),
    position VARCHAR(255),
    start_date DATE,
    end_date DATE,
    FOREIGN KEY (candidate_id) REFERENCES candidate_profiles(candidate_id)
);

-- Education records (multiple per candidate)
CREATE TABLE IF NOT EXISTS candidate_education (
    edu_id INT AUTO_INCREMENT PRIMARY KEY,
    candidate_id INT,
    school VARCHAR(255),
    university_rank INT,
    degree VARCHAR(255),
    start_year INT,
    end_year INT,
    FOREIGN KEY (candidate_id) REFERENCES candidate_profiles(candidate_id),
    UNIQUE KEY idx_unique_edu (candidate_id, school, start_year, end_year)
);