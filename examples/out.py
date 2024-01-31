import ipyvuetify as v
import ipywidgets as widgets
import IPython.display

a = v.Container(
    class_="blue",
    children=[
        v.Html(
            tag="h1", 
            children=["Hello World"]
            )
    ]
)
b = v.Container(
    class_="green",
    children=[
        v.Html(
            tag="h1", 
            children=["Hello Planet"]
            )
    ]
)

# def test():
#     print("Hello World")

def display(w: widgets.Widget):
    IPython.display.display(w)

def show(w: widgets.Widget):
    w.show()

def hide(w: widgets.Widget):
    w.hide()


