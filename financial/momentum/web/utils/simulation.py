import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
from financial.momentum.simulationFactory import SimulationFactory

def prepare_simulation_outputs(start_year, end_year, universe, architecture, extra_info, num_assets, refuge, lang="es"):
    factory = SimulationFactory(start_year=start_year, end_year=end_year)
    factory.run_simulation(universe, architecture, extra_info, num_assets, refuge)

    stats = factory.all_statistics()
    plot_html = generate_simulation_plot(factory, lang=lang)

    return stats, plot_html

def generate_simulation_plot(factory, lang="es"):
    models = factory.cumulative_returns()
    clenow = factory.clenow_cumulative_returns()
    bench = factory.marketBenchmark()["cumulative_returns"]

    cutoff = None
    idx = pd.concat([models, clenow, bench], axis=1).dropna(how="all").index
    if getattr(factory, "end_year", None) == 2025:
        cutoff = "2025-06-30"
    last_date = cutoff if cutoff is not None else idx.max()
    first_date = idx.min()

    models = models.loc[:last_date]
    clenow = clenow.loc[:last_date]
    bench  = bench.loc[:last_date]

    if lang == "en":
        title = "Cumulative Portfolio Return vs Benchmarks"
        xaxis_title = "Date"
        yaxis_title = "Cumulative Return"
        label_6m = "6m"
        label_1y = "1y"
        label_3y = "3y"
        label_all = "All"
        legend = "Simulation"
    else:
        title = "Rentabilidad Acumulada de la Cartera vs Benchmarks"
        xaxis_title = "Fecha"
        yaxis_title = "Rentabilidad Acumulada"
        label_6m = "6m"
        label_1y = "1a"
        label_3y = "3a"
        label_all = "Todo"
        legend = "Simulación"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=models.index, y=models.values, name=legend, line=dict(color="green")))
    fig.add_trace(go.Scatter(x=clenow.index, y=clenow.values, name="Clenow", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="Benchmark", line=dict(color="red")))

    fig.update_layout(
        title=title,
        height=600,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis=dict(ticksuffix="%", rangemode="tozero"),
        xaxis=dict(
            range=[first_date, last_date],
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label=label_6m, step="month", stepmode="backward"),
                    dict(count=1, label=label_1y, step="year", stepmode="backward"),
                    dict(count=3, label=label_3y, step="year", stepmode="backward"),
                    dict(step="all", label=label_all)
                ])
            ),
            rangeslider=dict(visible=True, range=[first_date, last_date]),
            type="date"
        )
    )

    return pio.to_html(fig, full_html=False)
