from jinja2.runtime import LoopContext, Macro, Markup, Namespace, TemplateNotFound, TemplateReference, TemplateRuntimeError, Undefined, escape, identity, internalcode, markup_join, missing, str_join
name = 'components/button/confirm.jinja'

def root(context, missing=missing):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    l_0_confirm_href = resolve('confirm_href')
    l_0_confirm_turbo_method = resolve('confirm_turbo_method')
    l_0_confirm_action_type = resolve('confirm_action_type')
    l_0_confirm_name = resolve('confirm_name')
    try:
        t_1 = environment.filters['default']
    except KeyError:
        @internalcode
        def t_1(*unused):
            raise TemplateRuntimeError("No filter named 'default' found.")
    pass
    yield '<a\n  href="'
    yield escape((undefined(name='confirm_href') if l_0_confirm_href is missing else l_0_confirm_href))
    yield '"\n  class="action_button"\n  data-turbo="true"\n  data-turbo-method="'
    yield escape(t_1((undefined(name='confirm_turbo_method') if l_0_confirm_turbo_method is missing else l_0_confirm_turbo_method), 'delete'))
    yield '"\n  data-action-type="confirm_'
    yield escape(t_1((undefined(name='confirm_action_type') if l_0_confirm_action_type is missing else l_0_confirm_action_type), 'delete'))
    yield '"\n  data-testid="confirm-action"\n>'
    yield escape(t_1((undefined(name='confirm_name') if l_0_confirm_name is missing else l_0_confirm_name), 'Confirm the action'))
    yield '</a>'

blocks = {}
debug_info = '2=22&5=24&6=26&8=28'