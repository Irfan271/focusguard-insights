from datetime import timedelta

import streamlit as st


def initialize_filter_state():
    """Initialize Streamlit session state used by dashboard filters."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    if "reset_filters" not in st.session_state:
        st.session_state.reset_filters = False

    if "selected_apps" not in st.session_state:
        st.session_state.selected_apps = []

    if "filter_version" not in st.session_state:
        st.session_state.filter_version = 0


def get_default_filters(df):
    """Return the default filter selections."""
    return {
        "selected_user": sorted(df["user_id"].unique())[0],
        "date_range_option": "All dates",
    }


def render_sidebar_filters(df):
    """
    Render sidebar filters and return the selected filter values.
    """
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    filter_version = st.session_state.filter_version

    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        st.caption(
            "Filter the dashboard by user, date range, and apps."
        )

        if st.button(
            "Reset filters",
            use_container_width=True,
        ):
            st.session_state.reset_filters = True
            st.rerun()

        st.divider()

        selected_user = st.selectbox(
            "User",
            sorted(df["user_id"].unique()),
            key=f"selected_user_{filter_version}",
            help="Choose whose app-usage data to display.",
        )

        date_range_option = st.selectbox(
            "Date range",
            [
                "All dates",
                "Last 7 days",
                "Last 30 days",
                "Custom range",
            ],
            key=f"date_range_option_{filter_version}",
            help=(
                "Choose a predefined period or select a custom range."
            ),
        )

        if date_range_option == "Custom range":
            selected_date_range = st.date_input(
                "Custom date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key=f"custom_date_range_{filter_version}",
                help=(
                    "Select the start and end dates to analyze."
                ),
            )
        else:
            selected_date_range = (min_date, max_date)

        selected_apps = render_app_filters(
            df,
            selected_user,
            filter_version,
        )

        render_active_filter_summary(
            selected_user,
            date_range_option,
            selected_date_range,
            selected_apps,
        )

        render_dataset_summary(df)

    return {
        "selected_user": selected_user,
        "date_range_option": date_range_option,
        "selected_date_range": selected_date_range,
        "selected_apps": selected_apps,
    }


def render_app_filters(df, selected_user, filter_version):
    """
    Render a searchable, category-browsable app selector.

    Apps from multiple categories can remain selected.
    """
    user_apps_df = df[df["user_id"] == selected_user]

    available_apps = sorted(
        user_apps_df["app_name"].unique()
    )

    # Remove apps that no longer exist for this user.
    st.session_state.selected_apps = [
        app
        for app in st.session_state.selected_apps
        if app in available_apps
    ]

    st.markdown("#### App selection")

    app_search = st.text_input(
        "Search apps",
        placeholder="Search by app name...",
        key=f"app_search_{filter_version}",
        help="Type to search for an app.",
    )

    app_category_view = st.selectbox(
        "Browse apps by category",
        ["All categories"]
        + sorted(user_apps_df["category"].unique()),
        key=f"app_category_view_{filter_version}",
        help=(
            "Choose a category to narrow the app list. Apps selected "
            "from other categories remain selected."
        ),
    )

    matching_apps_df = user_apps_df.copy()

    if app_search:
        matching_apps_df = matching_apps_df[
            matching_apps_df["app_name"].str.contains(
                app_search,
                case=False,
                na=False,
            )
        ]

    if app_category_view != "All categories":
        matching_apps_df = matching_apps_df[
            matching_apps_df["category"] == app_category_view
        ]

    matching_apps = sorted(
        matching_apps_df["app_name"].unique()
    )

    if matching_apps:
        if app_search:
            render_flat_app_checkboxes(
                matching_apps,
                filter_version,
            )
        else:
            render_grouped_app_checkboxes(
                matching_apps_df,
                filter_version,
            )
    else:
        st.info(
            "No apps match your search or category view."
        )

    selected_apps = st.session_state.selected_apps

    if selected_apps:
        st.caption(
            f"**Selected apps:** {', '.join(selected_apps)}"
        )
    else:
        st.caption("**Selected apps:** All apps")

    return selected_apps


def render_flat_app_checkboxes(apps, filter_version):
    """Render matching apps as a flat list during search."""
    for app in apps:
        render_app_checkbox(app, filter_version)


def render_grouped_app_checkboxes(
    matching_apps_df,
    filter_version,
):
    """Render apps grouped by category using expanders."""
    for app_category in sorted(
        matching_apps_df["category"].unique()
    ):
        category_apps = sorted(
            matching_apps_df[
                matching_apps_df["category"] == app_category
            ]["app_name"].unique()
        )

        with st.expander(
            f"{app_category} ({len(category_apps)})"
        ):
            for app in category_apps:
                render_app_checkbox(
                    app,
                    filter_version,
                )


def render_app_checkbox(app, filter_version):
    """
    Render one app checkbox.

    The checkbox updates selected_apps through an on_change callback,
    which avoids stale selection state.
    """
    checkbox_key = (
        f"app_checkbox_{filter_version}_{app}"
    )

    st.checkbox(
        app,
        value=app in st.session_state.selected_apps,
        key=checkbox_key,
        on_change=handle_app_checkbox_change,
        args=(app, checkbox_key),
    )


def handle_app_checkbox_change(app, checkbox_key):
    """
    Update the selected-apps list when a checkbox changes.
    """
    if st.session_state[checkbox_key]:
        if app not in st.session_state.selected_apps:
            st.session_state.selected_apps.append(app)
    else:
        if app in st.session_state.selected_apps:
            st.session_state.selected_apps.remove(app)


def render_active_filter_summary(
    selected_user,
    date_range_option,
    selected_date_range,
    selected_apps,
):
    """Show the filters currently applied to the dashboard."""
    st.divider()
    st.markdown("#### Active filters")

    if date_range_option == "Custom range":
        if len(selected_date_range) == 2:
            active_date_text = (
                f"{selected_date_range[0]} to "
                f"{selected_date_range[1]}"
            )
        else:
            active_date_text = "Select a complete date range"
    else:
        active_date_text = date_range_option

    if selected_apps:
        active_apps_text = f"{len(selected_apps)} selected"
    else:
        active_apps_text = "All apps"

    st.caption(f"**User:** {selected_user}")
    st.caption(f"**Dates:** {active_date_text}")
    st.caption(f"**Apps:** {active_apps_text}")


def render_dataset_summary(df):
    """Show high-level dataset statistics."""
    st.divider()
    st.markdown("#### Dataset")

    total_users = df["user_id"].nunique()
    total_apps = df["app_name"].nunique()
    total_records = len(df)

    st.metric("Users", total_users)
    st.metric("Apps", total_apps)
    st.metric("Records", f"{total_records:,}")

    st.divider()

    st.caption(
        "Data is synthetic and refreshed automatically for "
        "demonstration purposes."
    )


def apply_filters(df, filters):
    """
    Apply user, date-range, and app filters to the dataset.
    """
    selected_user = filters["selected_user"]
    date_range_option = filters["date_range_option"]
    selected_date_range = filters["selected_date_range"]
    selected_apps = filters["selected_apps"]

    filtered_df = df[df["user_id"] == selected_user]

    if date_range_option == "Last 7 days":
        end_date = df["date"].max().date()
        start_date = end_date - timedelta(days=6)

        filtered_df = filtered_df[
            filtered_df["date"].dt.date >= start_date
        ]
        filtered_df = filtered_df[
            filtered_df["date"].dt.date <= end_date
        ]

    elif date_range_option == "Last 30 days":
        end_date = df["date"].max().date()
        start_date = end_date - timedelta(days=29)

        filtered_df = filtered_df[
            filtered_df["date"].dt.date >= start_date
        ]
        filtered_df = filtered_df[
            filtered_df["date"].dt.date <= end_date
        ]

    elif date_range_option == "Custom range":
        if len(selected_date_range) == 2:
            start_date, end_date = selected_date_range

            filtered_df = filtered_df[
                filtered_df["date"].dt.date >= start_date
            ]
            filtered_df = filtered_df[
                filtered_df["date"].dt.date <= end_date
            ]

    if selected_apps:
        filtered_df = filtered_df[
            filtered_df["app_name"].isin(selected_apps)
        ]

    return filtered_df


def handle_filter_reset():
    """
    Reset all dashboard filters by creating fresh widget state.
    """
    if not st.session_state.reset_filters:
        return

    st.session_state.selected_apps = []

    # This forces Streamlit to create new filter widgets with defaults.
    st.session_state.filter_version += 1

    st.session_state.reset_filters = False