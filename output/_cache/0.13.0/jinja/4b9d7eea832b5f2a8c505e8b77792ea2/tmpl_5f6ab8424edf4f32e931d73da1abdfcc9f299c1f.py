from jinja2.runtime import LoopContext, Macro, Markup, Namespace, TemplateNotFound, TemplateReference, TemplateRuntimeError, Undefined, escape, identity, internalcode, markup_join, missing, str_join
name = 'components/node_content/index.jinja'

def root(context, missing=missing):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    l_0_sdoc_entity = resolve('sdoc_entity')
    l_0_requirement_style = resolve('requirement_style')
    l_0_user_requirement_style = l_0__no_title = l_0__has_multiline_fields = l_0__narrative_has_multiline_fields = l_0__narrative_has_no_multiline_fields = l_0_title_number = l_0_truncated_statement = missing
    try:
        t_1 = environment.filters['d']
    except KeyError:
        @internalcode
        def t_1(*unused):
            raise TemplateRuntimeError("No filter named 'd' found.")
    pass
    yield '\n\n\n\n\n'
    l_0_user_requirement_style = context.call(environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'get_requirement_style_mode'))
    context.vars['user_requirement_style'] = l_0_user_requirement_style
    context.exported_vars.add('user_requirement_style')
    yield '\n\n\n'
    l_0__no_title = (not environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'reserved_title'))
    context.vars['_no_title'] = l_0__no_title
    yield '\n'
    l_0__has_multiline_fields = context.call(environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'has_multiline_fields'))
    context.vars['_has_multiline_fields'] = l_0__has_multiline_fields
    yield '\n'
    l_0__narrative_has_multiline_fields = ((undefined(name='_has_multiline_fields') if l_0__has_multiline_fields is missing else l_0__has_multiline_fields) and ((undefined(name='user_requirement_style') if l_0_user_requirement_style is missing else l_0_user_requirement_style) == 'narrative'))
    context.vars['_narrative_has_multiline_fields'] = l_0__narrative_has_multiline_fields
    yield '\n'
    l_0__narrative_has_no_multiline_fields = ((not (undefined(name='_has_multiline_fields') if l_0__has_multiline_fields is missing else l_0__has_multiline_fields)) and ((undefined(name='user_requirement_style') if l_0_user_requirement_style is missing else l_0_user_requirement_style) == 'narrative'))
    context.vars['_narrative_has_no_multiline_fields'] = l_0__narrative_has_no_multiline_fields
    yield '\n\n<sdoc-node-content\n  node-view="'
    yield escape(t_1((undefined(name='requirement_style') if l_0_requirement_style is missing else l_0_requirement_style), context.call(environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'get_requirement_style_mode'))))
    yield '"\n  data-level="'
    yield escape(environment.getattr(environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'context'), 'title_number_string'))
    yield '"'
    if environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'reserved_status'):
        pass
        yield "\n    data-status='"
        yield escape(context.call(environment.getattr(environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'reserved_status'), 'lower')))
        yield "'"
    if (undefined(name='_narrative_has_no_multiline_fields') if l_0__narrative_has_no_multiline_fields is missing else l_0__narrative_has_no_multiline_fields):
        pass
        yield '\n    class="html2pdf4doc-no-break"'
    yield '\n  show-node-type-name="'
    yield escape(context.call(environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'get_node_type_string')))
    yield '"\n  data-testid="requirement-style-'
    yield escape(t_1((undefined(name='requirement_style') if l_0_requirement_style is missing else l_0_requirement_style), context.call(environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'get_requirement_style_mode'))))
    yield '"\n>\n  '
    l_0_title_number = True
    context.vars['title_number'] = l_0_title_number
    context.exported_vars.add('title_number')
    yield '\n  '
    l_0_truncated_statement = False
    context.vars['truncated_statement'] = l_0_truncated_statement
    context.exported_vars.add('truncated_statement')
    yield '\n  '
    template = environment.get_template('components/node_field/title/index.jinja', 'components/node_content/index.jinja')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    yield '\n  \n  \n\n  '
    if ((undefined(name='user_requirement_style') if l_0_user_requirement_style is missing else l_0_user_requirement_style) == 'narrative'):
        pass
        yield '\n    <sdoc-scope class="node_fields_group-secondary'
        if (undefined(name='_no_title') if l_0__no_title is missing else l_0__no_title):
            pass
            yield ' html2pdf4doc-no-hanging'
        yield '">\n      '
        template = environment.get_template('components/node_field/meta/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n      '
        template = environment.get_template('components/node_field/links/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n      '
        template = environment.get_template('components/node_field/files/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n    </sdoc-scope>\n    '
        if (undefined(name='_narrative_has_multiline_fields') if l_0__narrative_has_multiline_fields is missing else l_0__narrative_has_multiline_fields):
            pass
            yield '\n    <sdoc-scope class="node_fields_group-primary">\n      '
            template = environment.get_template('components/node_field/statement/index.jinja', 'components/node_content/index.jinja')
            gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
            try:
                for event in gen:
                    yield event
            finally: gen.close()
            yield '\n      '
            template = environment.get_template('components/node_field/rationale/index.jinja', 'components/node_content/index.jinja')
            gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
            try:
                for event in gen:
                    yield event
            finally: gen.close()
            yield '\n      '
            template = environment.get_template('components/node_field/comments/index.jinja', 'components/node_content/index.jinja')
            gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
            try:
                for event in gen:
                    yield event
            finally: gen.close()
            yield '\n      '
            template = environment.get_template('components/node_field/multiline/index.jinja', 'components/node_content/index.jinja')
            gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
            try:
                for event in gen:
                    yield event
            finally: gen.close()
            yield '\n    </sdoc-scope>\n    '
        yield '\n  '
    elif ((undefined(name='user_requirement_style') if l_0_user_requirement_style is missing else l_0_user_requirement_style) == 'plain'):
        pass
        yield '\n    '
        template = environment.get_template('components/node_field/statement/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n    '
        template = environment.get_template('components/node_field/rationale/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n    '
        template = environment.get_template('components/node_field/comments/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n    '
        template = environment.get_template('components/node_field/multiline/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n  '
    else:
        pass
        yield '\n    '
        template = environment.get_template('components/node_field/meta/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n    '
        template = environment.get_template('components/node_field/statement/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n    '
        template = environment.get_template('components/node_field/rationale/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n    '
        template = environment.get_template('components/node_field/comments/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n    '
        template = environment.get_template('components/node_field/multiline/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n    '
        template = environment.get_template('components/node_field/links/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n    '
        template = environment.get_template('components/node_field/files/index.jinja', 'components/node_content/index.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_multiline_fields': l_0__narrative_has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_no_title': l_0__no_title, 'title_number': l_0_title_number, 'truncated_statement': l_0_truncated_statement, 'user_requirement_style': l_0_user_requirement_style}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
        yield '\n  '
    yield '\n\n</sdoc-node-content>'

blocks = {}
debug_info = '18=21&35=25&36=28&37=31&38=34&41=37&42=39&43=41&44=44&46=46&49=50&50=52&52=54&53=58&54=62&62=69&63=72&64=76&65=83&66=90&68=97&70=100&71=107&72=114&73=121&76=129&77=132&78=139&79=146&80=153&82=163&83=170&84=177&85=184&86=191&87=198&88=205'