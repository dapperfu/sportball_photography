from jinja2.runtime import LoopContext, Macro, Markup, Namespace, TemplateNotFound, TemplateReference, TemplateRuntimeError, Undefined, escape, identity, internalcode, markup_join, missing, str_join
name = 'components/header/index.jinja'

def root(context, missing=missing):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    l_0_view_object = resolve('view_object')
    l_0_header__items = resolve('header__items')
    l_0_header__pagetype = resolve('header__pagetype')
    l_0_header__last = resolve('header__last')
    try:
        t_1 = environment.tests['defined']
    except KeyError:
        @internalcode
        def t_1(*unused):
            raise TemplateRuntimeError("No test named 'defined' found.")
    pass
    yield '<div class="header">\n\n  \n  '
    template = environment.get_template('_res/svg_ico16_project.jinja.html', 'components/header/index.jinja')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    yield '\n  <div\n    class="header__project_name"\n    title="'
    yield escape(environment.getattr(environment.getattr((undefined(name='view_object') if l_0_view_object is missing else l_0_view_object), 'project_config'), 'project_title'))
    yield '"\n  >'
    yield escape(environment.getattr(environment.getattr((undefined(name='view_object') if l_0_view_object is missing else l_0_view_object), 'project_config'), 'project_title'))
    yield '\n  </div>\n\n  '
    if t_1((undefined(name='header__items') if l_0_header__items is missing else l_0_header__items)):
        pass
        for l_1_item in (undefined(name='header__items') if l_0_header__items is missing else l_0_header__items):
            _loop_vars = {}
            pass
            yield '\n      '
            template = environment.get_template('_res/svg__separator.jinja.html', 'components/header/index.jinja')
            gen = template.root_render_func(template.new_context(context.get_all(), True, {'item': l_1_item}))
            try:
                for event in gen:
                    yield event
            finally: gen.close()
            yield '\n      '
            template = environment.get_or_select_template(l_1_item, 'components/header/index.jinja')
            gen = template.root_render_func(template.new_context(context.get_all(), True, {'item': l_1_item}))
            try:
                for event in gen:
                    yield event
            finally: gen.close()
            yield '\n    '
        l_1_item = missing
    if t_1((undefined(name='header__pagetype') if l_0_header__pagetype is missing else l_0_header__pagetype)):
        pass
        template = environment.get_template('_res/svg__separator.jinja.html', 'components/header/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n    '
        template = environment.get_template('components/header/header_pagetype.jinja', 'components/header/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
    if t_1((undefined(name='header__last') if l_0_header__last is missing else l_0_header__last)):
        pass
        template = environment.get_or_select_template((undefined(name='header__last') if l_0_header__last is missing else l_0_header__last), 'components/header/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
    yield '</div>'

blocks = {}
debug_info = '4=22&7=29&8=31&12=33&13=35&14=39&15=46&20=54&21=56&22=63&26=69&27=71'