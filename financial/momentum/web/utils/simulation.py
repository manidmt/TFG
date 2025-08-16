import plotly.graph_objs as go
import plotly.io as pio
from financial.momentum.simulationFactory import SimulationFactory

def prepare_simulation_outputs(start_year, end_year, universe, architecture, extra_info, num_assets, refuge, lang="es"):
    factory = SimulationFactory(start_year=start_year, end_year=end_year)
    simulation = factory.run_simulation(universe, architecture, extra_info, num_assets, refuge)
    
    stats = factory.portfolio_statistics()
    plot_html = generate_simulation_plot(factory, lang=lang)

    return stats, plot_html

def generate_simulation_plot(factory, lang="es"):
    df = factory.cumulative_returns()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.values,
        name="Rentabilidad acumulada",
        line=dict(color="green")
    ))

    if lang == "en":
        title = "Cumulative Portfolio Return"
        xaxis_title = "Date"
        yaxis_title = "Cumulative Return"
        label_6m = "6m"
        label_1y = "1y"
        label_3y = "3y"
        label_all = "All"
    else:
        title = "Rentabilidad Acumulada de la Cartera"
        xaxis_title = "Fecha"
        yaxis_title = "Rentabilidad Acumulada"
        label_6m = "6m"
        label_1y = "1a"
        label_3y = "3a"
        label_all = "Todo"

    fig.update_layout(
        title=title,
        height=600,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label=label_6m, step="month", stepmode="backward"),
                    dict(count=1, label=label_1y, step="year", stepmode="backward"),
                    dict(count=3, label=label_3y, step="year", stepmode="backward"),
                    dict(step="all", label=label_all)
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    return pio.to_html(fig, full_html=False)
