from pathlib import Path


Cancel
Comment

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DEFAULT_NEIGHBOR_LIMIT = 250


@st.cache_data
def load_projects():
    base_dir = Path(__file__).resolve().parent
    df = pd.read_csv(
        base_dir / "dataset" / "URA_enriched_with_99co_v3.csv",
        usecols=["Project Name", "latitude", "longitude"],
    )
    degrees = pd.read_csv(base_dir / "dataset" / "project_graph_node_degrees_within_5km.csv")

    projects = (
        df.dropna()
        .drop_duplicates(subset=["Project Name"])
        .merge(degrees, on="Project Name", how="left")
        .fillna({"edge_count": 0})
        .sort_values(["edge_count", "Project Name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    projects["edge_count"] = projects["edge_count"].astype(int)
    return projects


@st.cache_data
def load_edges():
    base_dir = Path(__file__).resolve().parent
    return pd.read_csv(
        base_dir / "dataset" / "project_graph_edges_within_5km.csv",
        usecols=[
            "source_project",
            "target_project",
            "source_latitude",
            "source_longitude",
            "target_latitude",
            "target_longitude",
            "distance_km",
        ],
    )


def build_neighbor_view(project_name, edges, projects, max_neighbors):
    source_edges = edges.loc[edges["source_project"] == project_name].copy()
    source_edges = source_edges.rename(
        columns={
            "target_project": "neighbor_project",
            "target_latitude": "neighbor_latitude",
            "target_longitude": "neighbor_longitude",
        }
    )
    source_edges["selected_latitude"] = source_edges["source_latitude"]
    source_edges["selected_longitude"] = source_edges["source_longitude"]

    target_edges = edges.loc[edges["target_project"] == project_name].copy()
    target_edges = target_edges.rename(
        columns={
            "source_project": "neighbor_project",
            "source_latitude": "neighbor_latitude",
            "source_longitude": "neighbor_longitude",
        }
    )
    target_edges["selected_latitude"] = target_edges["target_latitude"]
    target_edges["selected_longitude"] = target_edges["target_longitude"]

    neighbor_edges = pd.concat(
        [
            source_edges[
                [
                    "neighbor_project",
                    "neighbor_latitude",
                    "neighbor_longitude",
                    "selected_latitude",
                    "selected_longitude",
                    "distance_km",
                ]
            ],
            target_edges[
                [
                    "neighbor_project",
                    "neighbor_latitude",
                    "neighbor_longitude",
                    "selected_latitude",
                    "selected_longitude",
                    "distance_km",
                ]
            ],
        ],
        ignore_index=True,
    )

    neighbor_edges = (
        neighbor_edges.sort_values(["distance_km", "neighbor_project"])
        .head(max_neighbors)
        .merge(
            projects[["Project Name", "edge_count"]],
            left_on="neighbor_project",
            right_on="Project Name",
            how="left",
        )
        .drop(columns=["Project Name"])
        .rename(columns={"edge_count": "neighbor_edge_count"})
    )

    return neighbor_edges


def build_map_figure(selected_project, neighbor_edges, projects):
    selected_row = projects.loc[projects["Project Name"] == selected_project].iloc[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat=projects["latitude"],
            lon=projects["longitude"],
            mode="markers",
            marker={"size": 7, "color": "#6b7280", "opacity": 0.35},
            text=projects["Project Name"],
            customdata=projects[["edge_count"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Edge count: %{customdata[0]}<br>"
                "Lat: %{lat:.5f}<br>"
                "Lon: %{lon:.5f}<extra></extra>"
            ),
            name="All projects",
        )
    )

    if not neighbor_edges.empty:
        for row in neighbor_edges.itertuples(index=False):
            fig.add_trace(
                go.Scattermapbox(
                    lat=[row.selected_latitude, row.neighbor_latitude],
                    lon=[row.selected_longitude, row.neighbor_longitude],
                    mode="lines",
                    line={"width": 2, "color": "#ef4444"},
                    hovertemplate=(
                        f"{selected_project} -> {row.neighbor_project}<br>"
                        f"Distance: {row.distance_km:.3f} km<extra></extra>"
                    ),
                    name="Edge",
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scattermapbox(
                lat=neighbor_edges["neighbor_latitude"],
                lon=neighbor_edges["neighbor_longitude"],
                mode="markers",
                marker={"size": 10, "color": "#2563eb", "opacity": 0.9},
                text=neighbor_edges["neighbor_project"],
                customdata=neighbor_edges[["distance_km", "neighbor_edge_count"]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Distance to selected: %{customdata[0]:.3f} km<br>"
                    "Edge count: %{customdata[1]}<extra></extra>"
                ),
                name="Neighbors",
            )
        )

    fig.add_trace(
        go.Scattermapbox(
            lat=[selected_row["latitude"]],
            lon=[selected_row["longitude"]],
            mode="markers",
            marker={"size": 16, "color": "#f59e0b"},
            text=[selected_project],
            customdata=[[selected_row["edge_count"]]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Edge count: %{customdata[0]}<br>"
                "Lat: %{lat:.5f}<br>"
                "Lon: %{lon:.5f}<extra></extra>"
            ),
            name="Selected project",
        )
    )

    fig.update_layout(
        mapbox={
            "style": "carto-positron",
            "center": {"lat": float(selected_row["latitude"]), "lon": float(selected_row["longitude"])},
            "zoom": 11.5,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        height=700,
    )
    return fig


def main():
    st.set_page_config(page_title="PGIM Graph Demo", layout="wide")
    st.title("PGIM Project Graph Demo")
    st.caption("Visualize project nodes in Singapore and the <= 5 km edges around a selected project.")

    projects = load_projects()
    edges = load_edges()

    st.sidebar.header("Controls")
    project_names = projects["Project Name"].tolist()
    default_project = project_names[0] if project_names else None
    selected_project = st.sidebar.selectbox("Project", project_names, index=0 if default_project else None)
    max_neighbors = st.sidebar.slider("Neighbors to draw", min_value=10, max_value=500, value=DEFAULT_NEIGHBOR_LIMIT, step=10)
    show_neighbor_table = st.sidebar.checkbox("Show neighbor table", value=True)

    if not selected_project:
        st.warning("No projects found.")
        return

    neighbor_edges = build_neighbor_view(selected_project, edges, projects, max_neighbors)
    selected_stats = projects.loc[projects["Project Name"] == selected_project].iloc[0]

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Selected project edge count", int(selected_stats["edge_count"]))
    metric_2.metric("Neighbors shown", len(neighbor_edges))
    metric_3.metric(
        "Average displayed distance",
        f"{neighbor_edges['distance_km'].mean():.2f} km" if not neighbor_edges.empty else "0.00 km",
    )

    st.plotly_chart(
        build_map_figure(selected_project, neighbor_edges, projects),
        use_container_width=True,
    )

    stats_col, hist_col = st.columns([1.1, 1])

    with stats_col:
        st.subheader("Selected Project")
        st.write(
            pd.DataFrame(
                [
                    {
                        "Project Name": selected_stats["Project Name"],
                        "Latitude": selected_stats["latitude"],
                        "Longitude": selected_stats["longitude"],
                        "Edge Count": int(selected_stats["edge_count"]),
                    }
                ]
            )
        )

        if show_neighbor_table:
            st.subheader("Nearest Displayed Neighbors")
            st.dataframe(
                neighbor_edges[
                    ["neighbor_project", "distance_km", "neighbor_edge_count"]
                ].rename(
                    columns={
                        "neighbor_project": "Neighbor Project",
                        "distance_km": "Distance (km)",
                        "neighbor_edge_count": "Neighbor Edge Count",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with hist_col:
        st.subheader("Node Degree Distribution")
        hist = go.Figure(
            data=[
                go.Histogram(
                    x=projects["edge_count"],
                    nbinsx=40,
                    marker={"color": "#2563eb"},
                )
            ]
        )
        hist.update_layout(
            xaxis_title="Edge count",
            yaxis_title="Number of projects",
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            height=320,
        )
        st.plotly_chart(hist, use_container_width=True)


if __name__ == "__main__":
    main()