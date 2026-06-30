from flask import Blueprint, render_template, request, jsonify

from sql.DashboardStats import DashboardStats
from sql.CheckpointTrackMaster import CheckpointTrackMaster

bp_dashboard = Blueprint("dashboard", __name__)


@bp_dashboard.route("/dashboard")
def dashboard():
    stats_db = DashboardStats()

    summary = stats_db.summary()
    requests_per_day = stats_db.requests_per_day(14)
    checkpoints = CheckpointTrackMaster().get_all_checkpoints()[:15]
    inference_logs = stats_db.recent_inference_logs(20)
    event_type_filter = request.args.get("event_type") or None
    events = stats_db.recent_events(30, event_type=event_type_filter)
    event_types = stats_db.distinct_event_types()

    max_daily = max([c for _, c in requests_per_day], default=0) or 1

    return render_template(
        "dashboard.html",
        summary=summary,
        requests_per_day=requests_per_day,
        max_daily=max_daily,
        checkpoints=checkpoints,
        inference_logs=inference_logs,
        events=events,
        event_types=event_types,
        active_event_type=event_type_filter,
    )


@bp_dashboard.route("/dashboard/data")
def dashboard_data():
    """Lightweight JSON endpoint so the dashboard can auto-refresh without a full page reload."""
    stats_db = DashboardStats()
    return jsonify(
        summary=stats_db.summary(),
        requests_per_day=stats_db.requests_per_day(14),
    )
