-- ─────────────────────────────────────────────────────────────────
-- Useful analytical queries for the healthcare ML project.
-- ─────────────────────────────────────────────────────────────────


-- 1. Count of records per test result in cleaned table
SELECT test_results, COUNT(*) AS total
FROM cleaned_patients
GROUP BY test_results
ORDER BY total DESC;


-- 2. Class balance check (%)
SELECT
    test_results,
    COUNT(*) AS cnt,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM cleaned_patients
GROUP BY test_results;


-- 3. Average billing amount by admission type
SELECT
    admission_type,
    ROUND(AVG(billing_amount)::numeric, 2) AS avg_billing,
    COUNT(*) AS patients
FROM cleaned_patients
GROUP BY admission_type
ORDER BY avg_billing DESC;


-- 4. Most common medical conditions
SELECT medical_condition, COUNT(*) AS cases
FROM cleaned_patients
GROUP BY medical_condition
ORDER BY cases DESC
LIMIT 10;


-- 5. Latest model run summary
SELECT
    id,
    run_at,
    model_name,
    ROUND(accuracy::numeric, 4)        AS accuracy,
    ROUND(f1_score::numeric, 4)        AS f1,
    training_samples,
    test_samples,
    model_path
FROM model_runs
ORDER BY run_at DESC
LIMIT 5;


-- 6. Records loaded in the last 7 days
SELECT COUNT(*) AS recent_raw_records
FROM raw_patients
WHERE loaded_at >= NOW() - INTERVAL '7 days';


-- 7. Duplicate check in raw table (should return 0 rows)
SELECT name, date_of_admission, hospital, COUNT(*) AS cnt
FROM raw_patients
GROUP BY name, date_of_admission, hospital
HAVING COUNT(*) > 1;


-- 8. Gender distribution
SELECT gender, COUNT(*) AS cnt
FROM cleaned_patients
GROUP BY gender;


-- 9. Age bucket analysis
SELECT
    CASE
        WHEN age < 18  THEN 'Under 18'
        WHEN age < 35  THEN '18–34'
        WHEN age < 50  THEN '35–49'
        WHEN age < 65  THEN '50–64'
        ELSE '65+'
    END AS age_group,
    COUNT(*) AS patients,
    ROUND(AVG(billing_amount)::numeric, 2) AS avg_billing
FROM cleaned_patients
GROUP BY age_group
ORDER BY age_group;


-- 10. Length of stay vs test result
SELECT
    test_results,
    ROUND(AVG(length_of_stay)::numeric, 1) AS avg_los,
    MIN(length_of_stay) AS min_los,
    MAX(length_of_stay) AS max_los
FROM cleaned_patients
GROUP BY test_results;