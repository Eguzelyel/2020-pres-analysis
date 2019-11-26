from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextField, TextAreaField
from wtforms.validators import DataRequired


class MyForm(FlaskForm):
    class Meta:  # Ignoring CSRF security feature.
        csrf = False

    input_field = TextAreaField(label='Tweet Text:', render_kw={"rows": 11, "cols": 50}, id='input_field',
                                validators=[DataRequired()])
    submit = SubmitField('Submit')
