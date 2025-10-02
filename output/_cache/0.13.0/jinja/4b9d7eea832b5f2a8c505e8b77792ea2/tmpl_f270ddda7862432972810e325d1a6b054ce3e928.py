from jinja2.runtime import LoopContext, Macro, Markup, Namespace, TemplateNotFound, TemplateReference, TemplateRuntimeError, Undefined, escape, identity, internalcode, markup_join, missing, str_join
name = 'screens/document/document/frame_requirement_form.jinja'

def root(context, missing=missing):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    parent_template = None
    l_0_form_object = resolve('form_object')
    l_0_is_new_requirement = resolve('is_new_requirement')
    l_0_submit_url = resolve('submit_url')
    l_0_cancel_url = resolve('cancel_url')
    l_0_frame_id = missing
    pass
    parent_template = environment.get_template('components/form/index.jinja', 'screens/document/document/frame_requirement_form.jinja')
    for name, parent_block in parent_template.blocks.items():
        context.blocks.setdefault(name, []).append(parent_block)
    l_0_frame_id = markup_join(('article-', environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'requirement_mid'), ))
    context.vars['frame_id'] = l_0_frame_id
    context.exported_vars.add('frame_id')
    if (undefined(name='is_new_requirement') if l_0_is_new_requirement is missing else l_0_is_new_requirement):
        pass
        l_0_submit_url = '/actions/document/create_requirement'
        context.vars['submit_url'] = l_0_submit_url
        context.exported_vars.add('submit_url')
        l_0_cancel_url = markup_join(('/actions/document/cancel_new_requirement?requirement_mid=', environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'requirement_mid'), ))
        context.vars['cancel_url'] = l_0_cancel_url
        context.exported_vars.add('cancel_url')
    else:
        pass
        l_0_submit_url = '/actions/document/update_requirement'
        context.vars['submit_url'] = l_0_submit_url
        context.exported_vars.add('submit_url')
        l_0_cancel_url = markup_join(('/actions/document/cancel_edit_requirement?requirement_mid=', environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'requirement_mid'), ))
        context.vars['cancel_url'] = l_0_cancel_url
        context.exported_vars.add('cancel_url')
    yield from parent_template.root_render_func(context)

def block_form_content(context, missing=missing):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    _block_vars = {}
    l_0_is_new_requirement = resolve('is_new_requirement')
    l_0_reference_mid = resolve('reference_mid')
    l_0_whereto = resolve('whereto')
    l_0_form_object = resolve('form_object')
    l_0_namespace = resolve('namespace')
    l_0_text_field_row_context = l_0_requirement_mid = l_0_document_mid = l_0_element_type = l_0_relation_row_context = l_0_comment_field_row_context = missing
    try:
        t_1 = environment.filters['length']
    except KeyError:
        @internalcode
        def t_1(*unused):
            raise TemplateRuntimeError("No filter named 'length' found.")
    pass
    yield '\n\n  '
    if (undefined(name='is_new_requirement') if l_0_is_new_requirement is missing else l_0_is_new_requirement):
        pass
        yield '<input type="hidden" id="reference_mid" name="reference_mid" value="'
        yield escape((undefined(name='reference_mid') if l_0_reference_mid is missing else l_0_reference_mid))
        yield '" />\n    <input type="hidden" id="whereto" name="whereto" value="'
        yield escape((undefined(name='whereto') if l_0_whereto is missing else l_0_whereto))
        yield '" />'
    yield '<input type="hidden" id="requirement_mid" name="requirement_mid" value="'
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'requirement_mid'))
    yield '" />\n  <input type="hidden" id="document_mid" name="document_mid" value="'
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_mid'))
    yield '" />\n  <input type="hidden" id="context_document_mid" name="context_document_mid" value="'
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'context_document_mid'))
    yield '" />\n  <input type="hidden" id="element_type" name="element_type" value="'
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'element_type'))
    yield '" />\n\n\n<sdoc-tab-content id="Fields" active>\n  '
    l_0_text_field_row_context = context.call((undefined(name='namespace') if l_0_namespace is missing else l_0_namespace), _block_vars=_block_vars)
    _block_vars['text_field_row_context'] = l_0_text_field_row_context
    yield '\n  '
    l_0_requirement_mid = environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'requirement_mid')
    _block_vars['requirement_mid'] = l_0_requirement_mid
    yield '\n  '
    l_0_document_mid = environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_mid')
    _block_vars['document_mid'] = l_0_document_mid
    yield '\n  '
    l_0_element_type = environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'element_type')
    _block_vars['element_type'] = l_0_element_type
    yield '\n\n  '
    for l_1_error_ in context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'get_errors'), '_GENERAL_', _block_vars=_block_vars):
        _loop_vars = {}
        pass
        yield '<sdoc-form-row>\n    <sdoc-form-row-main>\n      <sdoc-form-error>\n        '
        yield escape(l_1_error_)
        yield '\n      </sdoc-form-error>\n    </sdoc-form-row-main>\n  </sdoc-form-row>'
    l_1_error_ = missing
    for l_1_field_values_ in context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'enumerate_fields'), multiline=False, _block_vars=_block_vars):
        _loop_vars = {}
        pass
        for l_2_field_ in l_1_field_values_:
            _loop_vars = {}
            pass
            if not isinstance(l_0_text_field_row_context, Namespace):
                raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
            l_0_text_field_row_context['errors'] = context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'get_errors'), environment.getattr(l_2_field_, 'field_name'), _loop_vars=_loop_vars)
            yield '\n      '
            if not isinstance(l_0_text_field_row_context, Namespace):
                raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
            l_0_text_field_row_context['field'] = l_2_field_
            yield '\n      '
            if not isinstance(l_0_text_field_row_context, Namespace):
                raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
            l_0_text_field_row_context['field_editable'] = True
            yield '\n      '
            if not isinstance(l_0_text_field_row_context, Namespace):
                raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
            l_0_text_field_row_context['field_type'] = 'singleline'
            yield '\n      '
            if not isinstance(l_0_text_field_row_context, Namespace):
                raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
            l_0_text_field_row_context['reference_mid'] = environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'requirement_mid')
            if (((environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'element_type') != 'TEXT') and (environment.getattr(l_2_field_, 'field_name') == 'UID')) and (environment.getattr(l_2_field_, 'field_value') == '')):
                pass
                yield '<turbo-frame id="uid_with_reset-'
                yield escape(environment.getattr((undefined(name='text_field_row_context') if l_0_text_field_row_context is missing else l_0_text_field_row_context), 'reference_mid'))
                yield '">\n          \n          '
                template = environment.get_template('components/form/row/row_uid_with_reset/frame.jinja', 'screens/document/document/frame_requirement_form.jinja')
                gen = template.root_render_func(template.new_context(context.get_all(), True, {'field_': l_2_field_, 'field_values_': l_1_field_values_, 'comment_field_row_context': l_0_comment_field_row_context, 'document_mid': l_0_document_mid, 'element_type': l_0_element_type, 'relation_row_context': l_0_relation_row_context, 'requirement_mid': l_0_requirement_mid, 'text_field_row_context': l_0_text_field_row_context}))
                try:
                    for event in gen:
                        yield event
                finally: gen.close()
                yield '\n        </turbo-frame>'
            else:
                pass
                template = environment.get_template('components/form/row/row_with_text_field.jinja', 'screens/document/document/frame_requirement_form.jinja')
                gen = template.root_render_func(template.new_context(context.get_all(), True, {'field_': l_2_field_, 'field_values_': l_1_field_values_, 'comment_field_row_context': l_0_comment_field_row_context, 'document_mid': l_0_document_mid, 'element_type': l_0_element_type, 'relation_row_context': l_0_relation_row_context, 'requirement_mid': l_0_requirement_mid, 'text_field_row_context': l_0_text_field_row_context}))
                try:
                    for event in gen:
                        yield event
                finally: gen.close()
        l_2_field_ = missing
    l_1_field_values_ = missing
    for l_1_field_values_ in context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'enumerate_fields'), multiline=True, _block_vars=_block_vars):
        _loop_vars = {}
        pass
        for l_2_field_ in l_1_field_values_:
            _loop_vars = {}
            pass
            if (environment.getattr(l_2_field_, 'field_name') != 'COMMENT'):
                pass
                if not isinstance(l_0_text_field_row_context, Namespace):
                    raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
                l_0_text_field_row_context['errors'] = context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'get_errors'), environment.getattr(l_2_field_, 'field_name'), _loop_vars=_loop_vars)
                yield '\n        '
                if not isinstance(l_0_text_field_row_context, Namespace):
                    raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
                l_0_text_field_row_context['field'] = l_2_field_
                yield '\n        '
                if not isinstance(l_0_text_field_row_context, Namespace):
                    raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
                l_0_text_field_row_context['field_editable'] = True
                yield '\n        '
                if not isinstance(l_0_text_field_row_context, Namespace):
                    raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
                l_0_text_field_row_context['field_type'] = 'multiline'
                yield '\n        '
                template = environment.get_template('components/form/row/row_with_text_field.jinja', 'screens/document/document/frame_requirement_form.jinja')
                gen = template.root_render_func(template.new_context(context.get_all(), True, {'field_': l_2_field_, 'field_values_': l_1_field_values_, 'comment_field_row_context': l_0_comment_field_row_context, 'document_mid': l_0_document_mid, 'element_type': l_0_element_type, 'relation_row_context': l_0_relation_row_context, 'requirement_mid': l_0_requirement_mid, 'text_field_row_context': l_0_text_field_row_context}))
                try:
                    for event in gen:
                        yield event
                finally: gen.close()
        l_2_field_ = missing
    l_1_field_values_ = missing
    yield '</sdoc-tab-content>\n\n\n<sdoc-tab-content id="Relations">\n  '
    l_0_relation_row_context = context.call((undefined(name='namespace') if l_0_namespace is missing else l_0_namespace), _block_vars=_block_vars)
    _block_vars['relation_row_context'] = l_0_relation_row_context
    for l_1_field_ in context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'enumerate_reference_fields'), _block_vars=_block_vars):
        _loop_vars = {}
        pass
        if not isinstance(l_0_relation_row_context, Namespace):
            raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
        l_0_relation_row_context['field'] = l_1_field_
        yield '\n    '
        if not isinstance(l_0_relation_row_context, Namespace):
            raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
        l_0_relation_row_context['errors'] = environment.getattr(l_1_field_, 'validation_messages')
        yield '\n    '
        if not isinstance(l_0_relation_row_context, Namespace):
            raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
        l_0_relation_row_context['relation_types'] = environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'relation_types')
        yield '\n    '
        if not isinstance(l_0_relation_row_context, Namespace):
            raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
        l_0_relation_row_context['form_object'] = (undefined(name='form_object') if l_0_form_object is missing else l_0_form_object)
        yield '\n    '
        template = environment.get_template('components/form/row/row_with_relation.jinja', 'screens/document/document/frame_requirement_form.jinja')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'field_': l_1_field_, 'comment_field_row_context': l_0_comment_field_row_context, 'document_mid': l_0_document_mid, 'element_type': l_0_element_type, 'relation_row_context': l_0_relation_row_context, 'requirement_mid': l_0_requirement_mid, 'text_field_row_context': l_0_text_field_row_context}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
    l_1_field_ = missing
    yield '<div id="requirement_'
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'requirement_mid'))
    yield '__new_relation"></div>\n  <sdoc-form-row>\n    <a\n      class="action_button"\n      href="/actions/document/new_relation?requirement_mid='
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'requirement_mid'))
    yield '&document_mid='
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_mid'))
    yield '&context_document_mid='
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'context_document_mid'))
    yield '&element_type='
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'element_type'))
    yield '"\n      data-turbo-action="replace"\n      data-turbo="true"\n      data-action-type="add_field"\n      data-testid="form-action-add-relation"\n    >'
    template = environment.get_template('_res/svg_ico16_add.jinja.html', 'screens/document/document/frame_requirement_form.jinja')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'comment_field_row_context': l_0_comment_field_row_context, 'document_mid': l_0_document_mid, 'element_type': l_0_element_type, 'relation_row_context': l_0_relation_row_context, 'requirement_mid': l_0_requirement_mid, 'text_field_row_context': l_0_text_field_row_context}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    yield ' Add relation</a>\n  </sdoc-form-row>\n</sdoc-tab-content>\n\n\n<sdoc-tab-content id="Comments">\n  '
    l_0_comment_field_row_context = context.call((undefined(name='namespace') if l_0_namespace is missing else l_0_namespace), _block_vars=_block_vars)
    _block_vars['comment_field_row_context'] = l_0_comment_field_row_context
    for l_1_field_values in context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'enumerate_fields'), multiline=True, _block_vars=_block_vars):
        _loop_vars = {}
        pass
        for l_2_field_ in l_1_field_values:
            _loop_vars = {}
            pass
            if (environment.getattr(l_2_field_, 'field_name') == 'COMMENT'):
                pass
                if (t_1(environment.getattr(l_2_field_, 'field_value')) > 0):
                    pass
                    if not isinstance(l_0_comment_field_row_context, Namespace):
                        raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
                    l_0_comment_field_row_context['field'] = l_2_field_
                    yield '\n          '
                    if not isinstance(l_0_comment_field_row_context, Namespace):
                        raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
                    l_0_comment_field_row_context['field_editable'] = True
                    yield '\n          '
                    if not isinstance(l_0_comment_field_row_context, Namespace):
                        raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
                    l_0_comment_field_row_context['errors'] = context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'get_errors'), environment.getattr(l_2_field_, 'field_name'), _loop_vars=_loop_vars)
                    yield '\n          '
                    template = environment.get_template('components/form/row/row_with_comment.jinja', 'screens/document/document/frame_requirement_form.jinja')
                    gen = template.root_render_func(template.new_context(context.get_all(), True, {'field_': l_2_field_, 'field_values': l_1_field_values, 'comment_field_row_context': l_0_comment_field_row_context, 'document_mid': l_0_document_mid, 'element_type': l_0_element_type, 'relation_row_context': l_0_relation_row_context, 'requirement_mid': l_0_requirement_mid, 'text_field_row_context': l_0_text_field_row_context}))
                    try:
                        for event in gen:
                            yield event
                    finally: gen.close()
        l_2_field_ = missing
    l_1_field_values = missing
    yield '\n  <div id="requirement_'
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'requirement_mid'))
    yield '__new_comment"></div>\n  <sdoc-form-row>\n    <a\n      class="action_button"\n      href="/actions/document/new_comment?requirement_mid='
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'requirement_mid'))
    yield '&document_mid='
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_mid'))
    yield '&context_document_mid='
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'context_document_mid'))
    yield '&element_type='
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'element_type'))
    yield '"\n      data-turbo-action="replace"\n      data-turbo="true"\n      data-action-type="add_field"\n      data-testid="form-action-add-comment"\n    >'
    template = environment.get_template('_res/svg_ico16_add.jinja.html', 'screens/document/document/frame_requirement_form.jinja')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'comment_field_row_context': l_0_comment_field_row_context, 'document_mid': l_0_document_mid, 'element_type': l_0_element_type, 'relation_row_context': l_0_relation_row_context, 'requirement_mid': l_0_requirement_mid, 'text_field_row_context': l_0_text_field_row_context}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    yield ' Add comment</a>\n  </sdoc-form-row>\n</sdoc-tab-content>\n\n'

blocks = {'form_content': block_form_content}
debug_info = '1=17&3=20&4=23&5=25&6=28&8=33&9=36&12=41&16=62&17=65&18=67&20=70&21=72&22=74&23=76&27=78&28=81&29=84&30=87&36=90&40=94&47=97&48=100&49=105&50=109&51=113&52=117&53=121&54=122&55=125&57=127&60=136&66=144&67=147&68=150&69=154&70=158&71=162&72=166&73=168&81=177&83=179&84=184&85=188&86=192&87=196&88=198&91=206&95=208&100=216&106=223&108=225&109=228&110=231&114=233&115=237&116=241&117=245&118=247&125=256&129=258&134=266'