USE climate_risk;
select * from vulnerability_scores;
select * from historical_baseline;
select * from disaster_events_clean;

-- Total Risk by State
SELECT 
    state,
    SUM(historical_vulnerability_index) AS total_state_risk
FROM vulnerability_scores
GROUP BY state
ORDER BY total_state_risk DESC;

-- Top Hazard Per State (Using Window Function)
SELECT *
FROM (
    SELECT 
        state,
        disaster_type,
        historical_vulnerability_index,
        RANK() OVER (
            PARTITION BY state 
            ORDER BY historical_vulnerability_index DESC
        ) AS hazard_rank
    FROM vulnerability_scores
) ranked
WHERE hazard_rank = 1;

-- Most Frequent Hazards
SELECT 
    state,
    disaster_type,
    avg_events_per_year
FROM historical_baseline
ORDER BY avg_events_per_year DESC;

-- Highest Financial Exposure
SELECT 
    state,
    disaster_type,
    avg_damage
FROM historical_baseline
ORDER BY avg_damage DESC;

-- Highest Human Impact
SELECT state,disaster_type,avg_deaths,avg_affected
from historical_baseline
ORDER BY avg_deaths DESC;

-- Risk Tier Classification
select state,disaster_type,historical_vulnerability_index,
case
	when historical_vulnerability_index >= 70 then "High Risk"
    when historical_vulnerability_index>= 40 then "Medium Risk"
    else "Low Risk"
end as risks
from vulnerability_scores;

-- Percentage Contribution to State Risk
SELECT 
    v.state,
    v.disaster_type,
    v.historical_vulnerability_index,
    ROUND(
        v.historical_vulnerability_index /
        SUM(v.historical_vulnerability_index) OVER (PARTITION BY v.state)
        * 100, 2
    ) AS percent_of_state_risk
FROM vulnerability_scores v;

-- How would you compute a rolling 5-year event count for each state–hazard combination using the disaster_events_clean table?
SELECT 
    state,
    disaster_type,
    start_year,
    COUNT(*) OVER (
        PARTITION BY state, disaster_type
        ORDER BY start_year
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rolling_5yr_events
FROM disaster_events_clean;

-- View: state_risk_summary
CREATE VIEW state_risk_summary AS
SELECT 
    state,
    SUM(historical_vulnerability_index) AS total_state_risk
FROM vulnerability_scores
GROUP BY state;

-- View: top_state_hazard
CREATE VIEW top_state_hazard AS
SELECT *
FROM (
    SELECT 
        state,
        disaster_type,
        historical_vulnerability_index,
        RANK() OVER (
            PARTITION BY state 
            ORDER BY historical_vulnerability_index DESC
        ) AS hazard_rank
    FROM vulnerability_scores
) ranked
WHERE hazard_rank = 1;

CREATE VIEW state_budget_summary AS
SELECT
    state,
    SUM(recommended_budget) AS total_required_budget,
    AVG(composite_prob) AS avg_occurrence_probability
FROM dynamic_disaster_budget
GROUP BY state;

CREATE VIEW high_priority_risks AS
SELECT *
FROM dynamic_disaster_budget
WHERE risk_tier IN ('High','Critical')
ORDER BY recommended_budget DESC;