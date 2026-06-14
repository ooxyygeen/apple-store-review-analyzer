from pydantic import BaseModel


class CollectResponse(BaseModel):
    app_id: int
    collected_at: str
    total_reviews_in_store: int
    reviews_collected: int


class StarCount(BaseModel):
    count: int
    percentage: float


class RatingStats(BaseModel):
    count: int
    mean: float
    median: float
    mode: int
    std_dev: float
    cumulative_positive_pct: float


class YearStats(BaseModel):
    count: int
    mean_rating: float


class Metrics(BaseModel):
    stats: RatingStats
    distribution: dict[str, StarCount]
    by_year: dict[str, YearStats]


class AnalysisResponse(BaseModel):
    app_id: int
    collected_at: str
    total_reviews_in_store: int
    metrics: Metrics
    actionable_insights: dict[str, str]